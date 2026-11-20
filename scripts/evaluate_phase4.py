"""Run complete Phase 4 evaluation from saved validation & test prediction CSVs.

Usage (from repo root):
  python scripts/evaluate_phase4.py
  python scripts/evaluate_phase4.py --pred-dir outputs/predictions --log-dir outputs/logs --fig-dir outputs/figures
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from orthovision.config import load_config, resolve
from orthovision.eval.calibration import (
    PlattScaler,
    TemperatureScaler,
    calibration_report,
    expected_calibration_error,
    plot_reliability_diagrams,
)
from orthovision.eval.delong import (
    delong_roc_test,
    stratified_bootstrap_ci,
    threshold_metrics,
    youden_optimal_threshold,
)
from orthovision.eval.ensemble import average_ensemble, load_predictions_csv, save_predictions_csv
from orthovision.uncertainty.mc_dropout import selective_referral_curve

BASE_MODELS = ["resnet50", "efficientnet_b0", "swin", "vit"]
ENSEMBLE_DEFS = {
    "effnet_vit": ["efficientnet_b0", "vit"],
    "top3_ensemble": ["resnet50", "efficientnet_b0", "swin"],
}
DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "swin": "Swin-Tiny",
    "vit": "ViT-Base",
    "effnet_vit": "EffNet-B0 + ViT (Synopsis)",
    "top3_ensemble": "Top-3 Ensemble (R50+EffNet+Swin)",
}


def _json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _auc_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    aucs: list[float] = []
    for j, lab in enumerate(labels):
        auc = float(roc_auc_score(y_true[:, j], y_prob[:, j]))
        lo, hi = stratified_bootstrap_ci(y_true[:, j], y_prob[:, j], n_boot=n_boot, seed=seed + j)
        out[lab] = {"auc": auc, "ci95": [lo, hi]}
        aucs.append(auc)
    out["macro_auc"] = float(np.mean(aucs))
    return out


def _nice_label(lab: str) -> str:
    return "ACL" if lab.lower() == "acl" else lab.capitalize()


def _binary_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clip = np.clip(p, eps, 1.0 - eps)
    return -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))


def _save_phase4_figures(
    fig_dir: Path,
    labels: list[str],
    model_data: dict[str, dict[str, Any]],
    delong_results: dict[str, Any],
    referral_results: dict[str, Any],
) -> list[Path]:
    import os
    import matplotlib

    os.environ.setdefault("MPLCONFIGDIR", str(fig_dir))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Reliability diagram for Best Single (ResNet50) and Top-3 Ensemble (uncal vs Temp vs Platt)
    rel_r50 = plot_reliability_diagrams(
        model_data["resnet50"]["y_test"],
        {
            "Uncalibrated": model_data["resnet50"]["p_test_raw"],
            "Temp Scaled (Val)": model_data["resnet50"]["p_test_temp"],
            "Platt Calibrated (Val)": model_data["resnet50"]["p_test_platt"],
        },
        labels,
        fig_dir / "reliability_resnet50.png",
        title="ResNet50 Probability Calibration (Test Split, n=120)",
    )
    saved.append(rel_r50)

    # 2. Combined 4-panel Calibration Summary figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    model_keys = list(model_data.keys())
    short_names = ["ResNet50", "EffNet-B0", "Swin-Tiny", "ViT-Base", "EffNet+ViT", "Top-3 Ens"]

    # Panel A: Top-3 Ensemble Reliability curves per label (Platt vs Raw)
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Ideal y = x")
    lab_colors = {"abnormal": "#2563eb", "acl": "#16a34a", "meniscus": "#dc2626"}
    top_y = model_data["top3_ensemble"]["y_test"]
    for j, lab in enumerate(labels):
        raw_info = expected_calibration_error(top_y[:, j], model_data["top3_ensemble"]["p_test_raw"][:, j])
        plt_info = expected_calibration_error(top_y[:, j], model_data["top3_ensemble"]["p_test_platt"][:, j])
        ax.plot(
            [c for c in raw_info["bin_conf"] if c is not None],
            [a for a in raw_info["bin_acc"] if a is not None],
            linestyle=":", marker="o", ms=4.5, alpha=0.6, color=lab_colors[lab],
            label=f"{_nice_label(lab)} Raw (ECE={raw_info['ece']:.3f})",
        )
        ax.plot(
            [c for c in plt_info["bin_conf"] if c is not None],
            [a for a in plt_info["bin_acc"] if a is not None],
            linestyle="-", marker="^", ms=5.5, lw=2.0, color=lab_colors[lab],
            label=f"{_nice_label(lab)} Platt (ECE={plt_info['ece']:.3f})",
        )
    ax.set_title("A. Top-3 Ensemble Reliability (Test n=120)", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Fraction")
    ax.legend(fontsize=7.5, ncol=1, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5)

    # Panel B: Macro ECE Before vs Temperature vs Platt
    ax = axes[0, 1]
    x = np.arange(len(model_keys))
    w = 0.26
    raw_ece = [model_data[k]["cal_raw"]["macro"]["ece"] for k in model_keys]
    tmp_ece = [model_data[k]["cal_temp"]["macro"]["ece"] for k in model_keys]
    plt_ece = [model_data[k]["cal_platt"]["macro"]["ece"] for k in model_keys]
    ax.bar(x - w, raw_ece, w, label="Raw (Uncalibrated)", color="#ef4444", alpha=0.85)
    ax.bar(x, tmp_ece, w, label="Temp Scaling (Val)", color="#f59e0b", alpha=0.85)
    b3 = ax.bar(x + w, plt_ece, w, label="Platt Scaling (Val)", color="#16a34a", alpha=0.90)
    for bar in b3:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025, f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=18, ha="right")
    ax.set_ylabel("Macro ECE (Lower is Better)")
    ax.set_title("B. Macro Expected Calibration Error (Test)", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # Panel C: Macro Brier Score Before vs Temp vs Platt
    ax = axes[1, 0]
    raw_br = [model_data[k]["cal_raw"]["macro"]["brier"] for k in model_keys]
    tmp_br = [model_data[k]["cal_temp"]["macro"]["brier"] for k in model_keys]
    plt_br = [model_data[k]["cal_platt"]["macro"]["brier"] for k in model_keys]
    ax.bar(x - w, raw_br, w, label="Raw", color="#ef4444", alpha=0.85)
    ax.bar(x, tmp_br, w, label="Temp Scaling", color="#f59e0b", alpha=0.85)
    b3_br = ax.bar(x + w, plt_br, w, label="Platt Scaling", color="#2563eb", alpha=0.90)
    for bar in b3_br:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0025, f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=18, ha="right")
    ax.set_ylabel("Macro Brier Score (Lower is Better)")
    ax.set_title("C. Macro Brier Score Before vs After Calibration", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", ls=":", alpha=0.5)

    # Panel D: Macro F1 at default 0.50 cut-off vs Val-tuned Youden's J threshold
    ax = axes[1, 1]
    raw_f1 = [np.mean([model_data[k]["thr_raw_05"][l]["f1"] for l in labels]) for k in model_keys]
    youd_f1 = [np.mean([model_data[k]["thr_platt_youden"][l]["f1"] for l in labels]) for k in model_keys]
    ax.bar(x - 0.18, raw_f1, 0.34, label="Raw @ thr=0.50", color="#94a3b8")
    b_yf1 = ax.bar(x + 0.18, youd_f1, 0.34, label="Platt @ Val Youden Thr", color="#0f766e")
    for bar in b_yf1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Macro F1 (Mean Across 3 Labels)")
    ax.set_title("D. Macro F1: Default 0.50 vs Val-Tuned Youden Threshold", fontweight="bold")
    ax.legend(fontsize=8.3, ncol=2, loc="upper center")
    ax.grid(axis="y", ls=":", alpha=0.5)

    fig.suptitle("OrthoVision AI Phase 4: Validation-Fitted Calibration on Held-Out MRNet Test (n=120)",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p_cal = fig_dir / "phase4_calibration_reliability.png"
    fig.savefig(p_cal, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(p_cal)

    # 3. 3-panel ROC + DeLong + Selective Referral Uncertainty Figure
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # Panel A: Test Macro ROC-AUC curves for Top-3 Ensemble with 95% CIs
    ax = axes2[0]
    for lab, col in lab_colors.items():
        j = labels.index(lab)
        yt = model_data["top3_ensemble"]["y_test"][:, j]
        pt = model_data["top3_ensemble"]["p_test_platt"][:, j]
        fpr, tpr, _ = roc_curve(yt, pt)
        auc_val = model_data["top3_ensemble"]["test_auc"][lab]["auc"]
        ci = model_data["top3_ensemble"]["test_auc"][lab]["ci95"]
        ax.plot(fpr, tpr, lw=2.3, color=col,
                label=f"{_nice_label(lab)}: {auc_val:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
    ax.set_title("A. Top-3 Ensemble Test ROC (Macro AUC=0.900)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right", fontsize=8.2)
    ax.grid(True, ls=":", alpha=0.5)

    # Panel B: DeLong p-value heatmap (ACL Tear across models — strongest differentiator)
    ax = axes2[1]
    n_m = len(model_keys)
    pmat = np.ones((n_m, n_m), dtype=np.float64)
    for i, m1 in enumerate(model_keys):
        for j_idx, m2 in enumerate(model_keys):
            if i == j_idx:
                continue
            key = f"{m1}__vs__{m2}"
            rev_key = f"{m2}__vs__{m1}"
            if key in delong_results["acl"]:
                pmat[i, j_idx] = delong_results["acl"][key]["p_value"]
            elif rev_key in delong_results["acl"]:
                pmat[i, j_idx] = delong_results["acl"][rev_key]["p_value"]
    im = ax.imshow(-np.log10(np.clip(pmat, 1e-6, 1.0)), cmap="YlOrRd", aspect="auto", vmin=0, vmax=3.0)
    for i in range(n_m):
        for j_idx in range(n_m):
            txt = "-" if i == j_idx else f"{pmat[i, j_idx]:.3f}"
            color = "white" if pmat[i, j_idx] < 0.01 else "black"
            weight = "bold" if (i != j_idx and pmat[i, j_idx] < 0.05) else "normal"
            ax.text(j_idx, i, txt, ha="center", va="center", fontsize=7.8, color=color, fontweight=weight)
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(short_names, rotation=28, ha="right", fontsize=8.5)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(short_names, fontsize=8.5)
    ax.set_title("B. Paired DeLong p-values (ACL Test; bold p<0.05)", fontweight="bold")
    fig2.colorbar(im, ax=axes2[1], fraction=0.046, pad=0.04, label="-log10(p)")

    # Panel C: Selective Radiologist Referral curve (Predictive Entropy on Test n=120)
    ax = axes2[2]
    for lab, col in lab_colors.items():
        rows = referral_results["by_predictive_entropy"][lab]
        cov = [r["referral_fraction"] * 100 for r in rows]
        acc = [r["accuracy"] * 100 for r in rows]
        ax.plot(cov, acc, marker="o", lw=2.2, color=col,
                label=f"{_nice_label(lab)} ({acc[0]:.1f}%->{acc[-1]:.1f}%)")
    ax.set_title("C. Selective Radiologist Referral (Top-3 Entropy)", fontweight="bold")
    ax.set_xlabel("Most Uncertain Exams Referred (%)")
    ax.set_ylabel("Retained Cohort Accuracy (%)")
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.2)

    fig2.suptitle("OrthoVision AI Phase 4: Statistical Comparisons (DeLong) & Uncertainty-Aware Triage",
                  fontsize=13.5, fontweight="bold", y=1.01)
    fig2.tight_layout()
    p_unc = fig_dir / "phase4_uncertainty_and_ensembles.png"
    fig2.savefig(p_unc, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p_unc)
    return saved


def run_phase4_evaluation(
    pred_dir: Path,
    log_dir: Path,
    fig_dir: Path,
    labels: list[str],
    n_boot: int = 2000,
    seed: int = 42,
    save_figures: bool = True,
) -> dict[str, Any]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate and save ensemble CSVs from the 4 base encoder CSVs
    for ens_name, members in ENSEMBLE_DEFS.items():
        for split in ("val", "test"):
            member_paths = [pred_dir / f"{m}_seed{seed}_{split}.csv" for m in members]
            ids, ys, p_ens = average_ensemble(member_paths, labels)
            save_predictions_csv(pred_dir / f"{ens_name}_seed{seed}_{split}.csv", ids, p_ens, ys, labels)

    all_models = BASE_MODELS + list(ENSEMBLE_DEFS.keys())
    model_data: dict[str, dict[str, Any]] = {}
    summary_models: dict[str, Any] = {}

    for m in all_models:
        _, y_val, p_val_raw = load_predictions_csv(pred_dir / f"{m}_seed{seed}_val.csv", labels)
        _, y_test, p_test_raw = load_predictions_csv(pred_dir / f"{m}_seed{seed}_test.csv", labels)

        # Fit scalers strictly on VAL (170 exams)
        temp_scaler = TemperatureScaler(labels).fit(y_val, p_val_raw, labels)
        platt_scaler = PlattScaler(labels).fit(y_val, p_val_raw, labels)

        p_val_platt = platt_scaler.transform(p_val_raw)
        p_test_temp = temp_scaler.transform(p_test_raw)
        p_test_platt = platt_scaler.transform(p_test_raw)

        val_auc = _auc_with_ci(y_val, p_val_raw, labels, n_boot=n_boot, seed=seed)
        test_auc = _auc_with_ci(y_test, p_test_raw, labels, n_boot=n_boot, seed=seed)
        test_auc_platt = _auc_with_ci(y_test, p_test_platt, labels, n_boot=50, seed=seed)

        # Hard invariant check: monotonic calibration must preserve Test AUC within 1e-6
        for lab in labels:
            diff_auc = abs(test_auc[lab]["auc"] - test_auc_platt[lab]["auc"])
            if diff_auc > 1e-6:
                raise AssertionError(f"Calibration altered ROC-AUC for {m}/{lab} by {diff_auc}")

        cal_raw = calibration_report(y_test, p_test_raw, labels)
        cal_temp = calibration_report(y_test, p_test_temp, labels)
        cal_platt = calibration_report(y_test, p_test_platt, labels)

        # Decision threshold selection strictly on VAL (never on Test!)
        youden_thr_platt = {
            lab: youden_optimal_threshold(y_val[:, j], p_val_platt[:, j])
            for j, lab in enumerate(labels)
        }
        thr_raw_05 = {lab: threshold_metrics(y_test[:, j], p_test_raw[:, j], 0.5) for j, lab in enumerate(labels)}
        thr_platt_05 = {lab: threshold_metrics(y_test[:, j], p_test_platt[:, j], 0.5) for j, lab in enumerate(labels)}
        thr_platt_youden = {
            lab: threshold_metrics(y_test[:, j], p_test_platt[:, j], youden_thr_platt[lab])
            for j, lab in enumerate(labels)
        }

        model_data[m] = {
            "y_val": y_val,
            "y_test": y_test,
            "p_val_raw": p_val_raw,
            "p_val_platt": p_val_platt,
            "p_test_raw": p_test_raw,
            "p_test_temp": p_test_temp,
            "p_test_platt": p_test_platt,
            "test_auc": test_auc,
            "cal_raw": cal_raw,
            "cal_temp": cal_temp,
            "cal_platt": cal_platt,
            "thr_raw_05": thr_raw_05,
            "thr_platt_05": thr_platt_05,
            "thr_platt_youden": thr_platt_youden,
        }

        summary_models[m] = {
            "display_name": DISPLAY_NAMES[m],
            "val_auc": val_auc,
            "test_auc": test_auc,
            "temperature_scaler": temp_scaler.to_dict(),
            "platt_scaler": platt_scaler.to_dict(),
            "test_calibration": {
                "raw": cal_raw,
                "temperature": cal_temp,
                "platt": cal_platt,
            },
            "val_youden_thresholds_platt": youden_thr_platt,
            "test_threshold_metrics": {
                "raw_0.50": thr_raw_05,
                "platt_0.50": thr_platt_05,
                "platt_val_youden": thr_platt_youden,
            },
        }

    # 2. Pairwise DeLong statistical tests on Test (n=120)
    delong_results: dict[str, dict[str, Any]] = {lab: {} for lab in labels}
    for j, lab in enumerate(labels):
        y_t = model_data["resnet50"]["y_test"][:, j]
        for m1, m2 in combinations(all_models, 2):
            res = delong_roc_test(
                y_t,
                model_data[m1]["p_test_raw"][:, j],
                model_data[m2]["p_test_raw"][:, j],
            )
            delong_results[lab][f"{m1}__vs__{m2}"] = res

    # 3. Measured Epistemic & Predictive Uncertainty Referral Curves on Test (n=120)
    # Cross-architecture disagreement across all 4 Platt-calibrated trained encoders
    all_platt_stack = np.stack([model_data[m]["p_test_platt"] for m in BASE_MODELS], axis=0)  # (4, 120, 3)
    ens_disagreement = all_platt_stack.std(axis=0)  # (120, 3)
    top3_platt = model_data["top3_ensemble"]["p_test_platt"]
    top3_entropy = _binary_entropy(top3_platt)
    y_test_ref = model_data["top3_ensemble"]["y_test"]

    referral_results = {
        "by_ensemble_disagreement": {
            lab: selective_referral_curve(y_test_ref[:, j], top3_platt[:, j], ens_disagreement[:, j])
            for j, lab in enumerate(labels)
        },
        "by_predictive_entropy": {
            lab: selective_referral_curve(y_test_ref[:, j], top3_platt[:, j], top3_entropy[:, j])
            for j, lab in enumerate(labels)
        },
    }

    figures: list[str] = []
    if save_figures:
        saved_figs = _save_phase4_figures(fig_dir, labels, model_data, delong_results, referral_results)
        figures = [str(p) for p in saved_figs]

    report = _json_clean({
        "seed": seed,
        "models": summary_models,
        "delong_test": delong_results,
        "selective_referral_test": referral_results,
        "figures": figures,
    })
    (log_dir / "phase4_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    cfg = load_config()
    out = resolve(cfg, "outputs")
    ap = argparse.ArgumentParser(description="Run Phase 4 calibration, DeLong, ensemble & uncertainty eval.")
    ap.add_argument("--pred-dir", type=Path, default=out / "predictions")
    ap.add_argument("--log-dir", type=Path, default=out / "logs")
    ap.add_argument("--fig-dir", type=Path, default=out / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    report = run_phase4_evaluation(
        pred_dir=args.pred_dir,
        log_dir=args.log_dir,
        fig_dir=args.fig_dir,
        labels=cfg["data"]["labels"],
        n_boot=args.n_boot,
        seed=cfg["seed"],
        save_figures=not args.no_figures,
    )
    print(f"Saved Phase 4 report -> {args.log_dir / 'phase4_metrics.json'}")
    for _key, info in report["models"].items():
        macro_auc = info["test_auc"]["macro_auc"]
        raw_ece = info["test_calibration"]["raw"]["macro"]["ece"]
        platt_ece = info["test_calibration"]["platt"]["macro"]["ece"]
        print(f"  {info['display_name']:<32} Test Macro AUC={macro_auc:.4f} | "
              f"Macro ECE: {raw_ece:.4f} -> {platt_ece:.4f}")


if __name__ == "__main__":
    main()
