"""Modular data, inference-summary, calibration, and triage backend for OrthoVision AI.

Designed for two-stage evolution without changing the Streamlit UI (`app/app.py`):
- Current (Phase 4 Demo): reads locked Val/Test prediction CSVs, Phase 4 metrics JSON,
  training logs, and optional local MRNet `.npy` slices (pure CPU, no `torch` import).
- Future (Phases 5-6): drop Phase 5 Grad-CAM overlay generation into
  `load_gradcam_overlays(...)` / `ExamAnalysisResult.gradcam_overlays` and Phase 6 LLM
  clinical narrative synthesis into `generate_clinical_summary(...)`.

Research prototype only — not for clinical diagnosis.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orthovision.eval.calibration import calibration_report, prob_to_logit, sigmoid  # noqa: E402
from orthovision.eval.delong import threshold_metrics, youden_optimal_threshold  # noqa: E402
from orthovision.uncertainty.mc_dropout import (  # noqa: E402
    _binary_entropy,
    selective_referral_curve,
)

LABELS: tuple[str, ...] = ("abnormal", "acl", "meniscus")
LABEL_DISPLAY: dict[str, str] = {
    "abnormal": "Abnormal",
    "acl": "ACL Tear",
    "meniscus": "Meniscus Tear",
}
PLANES: tuple[str, ...] = ("axial", "coronal", "sagittal")
BASE_ENCODERS: tuple[str, ...] = ("resnet50", "efficientnet_b0", "swin", "vit")
ALL_MODELS: tuple[str, ...] = (
    "top3_ensemble",
    "resnet50",
    "swin",
    "efficientnet_b0",
    "effnet_vit",
    "vit",
)
MODEL_DISPLAY: dict[str, str] = {
    "top3_ensemble": "Top-3 Ensemble (R50 + EffNet-B0 + Swin)",
    "resnet50": "ResNet50",
    "swin": "Swin-Tiny",
    "efficientnet_b0": "EfficientNet-B0",
    "effnet_vit": "EffNet-B0 + ViT (Synopsis)",
    "vit": "ViT-Base",
}
MODEL_SHORT: dict[str, str] = {
    "top3_ensemble": "Top-3 Ensemble",
    "resnet50": "ResNet50",
    "swin": "Swin-Tiny",
    "efficientnet_b0": "EfficientNet-B0",
    "effnet_vit": "EffNet+ViT",
    "vit": "ViT-Base",
}
REFERRAL_FRACTIONS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


class MissingOutputsError(FileNotFoundError):
    """Raised when required Phase 3/4 output files are not present locally."""


@dataclass(frozen=True)
class Phase4Bundle:
    """In-memory container holding all Phase 3/4 logs and prediction tables."""

    seed: int
    metrics: dict[str, Any]
    val_dfs: dict[str, pd.DataFrame]
    test_dfs: dict[str, pd.DataFrame]
    train_logs: dict[str, pd.DataFrame]
    exam_ids: tuple[int, ...]


@dataclass
class ExamAnalysisResult:
    """Structured per-exam research output consumed by the Streamlit explorer."""

    exam_id: int
    model_key: str
    model_name: str
    calibration_mode: str
    threshold_mode: str
    probs: dict[str, float]
    raw_probs: dict[str, float]
    thresholds: dict[str, float]
    predictions: dict[str, int]
    ground_truth: dict[str, int]
    per_encoder_probs: dict[str, dict[str, float]]
    predictive_entropy: dict[str, float]
    encoder_disagreement: dict[str, float]
    triage_referred: bool
    triage_percentile: float
    triage_reason: str
    gradcam_overlays: dict[str, Any] | None
    clinical_summary: str


def _require_files(paths: list[Path], repo_root: Path) -> None:
    missing = [str(p.relative_to(repo_root)) for p in paths if not p.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise MissingOutputsError(
            "Missing Phase 3/4 output files locally (outputs/ is git-ignored):\n"
            f"  - {joined}\n\n"
            "To fix this:\n"
            "  1. Download prediction CSVs + logs from Google Drive ('OrthoVision Backups') "
            "into outputs/predictions/ and outputs/logs/, then\n"
            "  2. Run: python scripts/evaluate_phase4.py"
        )


def load_phase4_bundle(
    repo_root: Path = REPO_ROOT,
    seed: int = 42,
) -> Phase4Bundle:
    """Load Phase 4 evaluation metrics, 12 prediction CSVs, and 4 encoder training logs."""
    pred_dir = repo_root / "outputs" / "predictions"
    log_dir = repo_root / "outputs" / "logs"
    metrics_path = log_dir / "phase4_metrics.json"

    required = [metrics_path]
    for model_key in ALL_MODELS:
        required.append(pred_dir / f"{model_key}_seed{seed}_val.csv")
        required.append(pred_dir / f"{model_key}_seed{seed}_test.csv")
    for enc in BASE_ENCODERS:
        required.append(log_dir / f"{enc}_seed{seed}" / "log.csv")
    _require_files(required, repo_root)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    val_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for model_key in ALL_MODELS:
        val_dfs[model_key] = pd.read_csv(pred_dir / f"{model_key}_seed{seed}_val.csv")
        test_dfs[model_key] = pd.read_csv(pred_dir / f"{model_key}_seed{seed}_test.csv")

    train_logs: dict[str, pd.DataFrame] = {
        enc: pd.read_csv(log_dir / f"{enc}_seed{seed}" / "log.csv")
        for enc in BASE_ENCODERS
    }
    ref_ids = tuple(int(x) for x in test_dfs[ALL_MODELS[0]]["id"].tolist())
    return Phase4Bundle(
        seed=seed,
        metrics=metrics,
        val_dfs=val_dfs,
        test_dfs=test_dfs,
        train_logs=train_logs,
        exam_ids=ref_ids,
    )


def _df_prob_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"p_{lab}" for lab in LABELS]].to_numpy(dtype=np.float64)


def _df_label_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"y_{lab}" for lab in LABELS]].to_numpy(dtype=np.int64)


def apply_calibration(
    bundle: Phase4Bundle,
    model_key: str,
    probs: np.ndarray,
    calibration_mode: str = "platt",
) -> np.ndarray:
    """Apply Validation-fitted Temperature or Platt scaling (or return raw probabilities)."""
    arr = np.asarray(probs, dtype=np.float64)
    if calibration_mode == "raw":
        return arr.copy()

    model_meta = bundle.metrics["models"][model_key]
    logits = prob_to_logit(arr)
    out = np.empty_like(arr, dtype=np.float64)

    if calibration_mode == "temperature":
        temps = model_meta["temperature_scaler"]["temperatures"]
        for j, lab in enumerate(LABELS):
            out[:, j] = sigmoid(logits[:, j] / float(temps[lab]))
        return out

    if calibration_mode == "platt":
        params = model_meta["platt_scaler"]["params"]
        for j, lab in enumerate(LABELS):
            a = float(params[lab]["a"])
            b = float(params[lab]["b"])
            out[:, j] = sigmoid(a * logits[:, j] + b)
        return out

    raise ValueError(f"Unsupported calibration_mode: {calibration_mode!r}")


def get_thresholds(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str = "platt",
    threshold_mode: str = "val_youden",
) -> dict[str, float]:
    """Return per-label decision thresholds using strictly Validation data or fixed 0.50."""
    if threshold_mode == "fixed_0.50":
        return {lab: 0.50 for lab in LABELS}
    if threshold_mode != "val_youden":
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode!r}")

    if calibration_mode == "platt":
        saved = bundle.metrics["models"][model_key]["val_youden_thresholds_platt"]
        return {lab: float(saved[lab]) for lab in LABELS}

    val_df = bundle.val_dfs[model_key]
    y_val = _df_label_matrix(val_df)
    p_val_cal = apply_calibration(bundle, model_key, _df_prob_matrix(val_df), calibration_mode)
    return {
        lab: float(youden_optimal_threshold(y_val[:, j], p_val_cal[:, j]))
        for j, lab in enumerate(LABELS)
    }


def _compute_uncertainty_matrices(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(p_test_cal, predictive_entropy, encoder_disagreement)` each of shape `(N, 3)`."""
    p_test_raw = _df_prob_matrix(bundle.test_dfs[model_key])
    p_test_cal = apply_calibration(bundle, model_key, p_test_raw, calibration_mode)
    pred_entropy = _binary_entropy(p_test_cal)

    enc_stack = np.stack(
        [
            apply_calibration(
                bundle,
                enc,
                _df_prob_matrix(bundle.test_dfs[enc]),
                calibration_mode,
            )
            for enc in BASE_ENCODERS
        ],
        axis=0,
    )
    disagreement = enc_stack.std(axis=0, ddof=0)
    return p_test_cal, pred_entropy, disagreement


def _triage_status_for_exam(
    exam_idx: int,
    macro_scores: np.ndarray,
    referral_rate: float,
    uncertainty_metric: str,
    per_label_entropy: dict[str, float],
    per_label_disagreement: dict[str, float],
) -> tuple[bool, float, str]:
    n_total = len(macro_scores)
    target_val = float(macro_scores[exam_idx])
    percentile = float(100.0 * (macro_scores <= target_val).sum() / n_total)

    n_refer = int(round(float(referral_rate) * n_total))
    desc_order = np.argsort(-macro_scores, kind="mergesort")
    referred_indices = set(int(i) for i in desc_order[:n_refer]) if n_refer > 0 else set()
    is_referred = exam_idx in referred_indices

    metric_map = (
        per_label_disagreement if uncertainty_metric == "encoder_disagreement" else per_label_entropy
    )
    top_label = max(LABELS, key=lambda lab: metric_map[lab])
    metric_label = (
        "4-encoder disagreement"
        if uncertainty_metric == "encoder_disagreement"
        else "predictive entropy"
    )

    if n_refer == 0:
        reason = (
            f"Referral budget is set to 0%; highest uncertainty on {LABEL_DISPLAY[top_label]} "
            f"({metric_label}: {metric_map[top_label]:.3f}, {percentile:.0f}th percentile)."
        )
    elif is_referred:
        reason = (
            f"Ranked at {percentile:.0f}th uncertainty percentile (within top {referral_rate * 100:.0f}% "
            f"referral budget), driven mainly by {LABEL_DISPLAY[top_label]} {metric_label} "
            f"({metric_map[top_label]:.3f})."
        )
    else:
        reason = (
            f"Ranked at {percentile:.0f}th uncertainty percentile (below the top "
            f"{referral_rate * 100:.0f}% referral cutoff); highest signal uncertainty is on "
            f"{LABEL_DISPLAY[top_label]} ({metric_map[top_label]:.3f})."
        )
    return is_referred, percentile, reason


def load_gradcam_overlays(exam_id: int) -> dict[str, Any] | None:
    """Phase 5 extension hook: return Grad-CAM heatmaps & region scores when implemented.

    In the Phase 4 interactive demo this returns `None`, signaling the UI to display the
    Phase 5 plug-in ready card (or raw MRI planes if local `.npy` volumes exist).
    """
    _ = exam_id
    return None


def generate_clinical_summary(result_dict: ExamAnalysisResult | dict[str, Any]) -> str:
    """Build a structured plain-English research summary (Phase 6 LLM extension hook).

    Replace the template formatting inside this function with the Phase 6 LLM synthesis
    pipeline once structured Phase 5 Grad-CAM region evidence is connected.
    """
    data = asdict(result_dict) if isinstance(result_dict, ExamAnalysisResult) else result_dict
    probs = data["probs"]
    preds = data["predictions"]
    thresholds = data["thresholds"]
    ent = data["predictive_entropy"]
    dis = data["encoder_disagreement"]

    positives = [LABEL_DISPLAY[lab] for lab in LABELS if int(preds[lab]) == 1]
    finding_text = (
        f"Elevated calibrated risk detected for **{', '.join(positives)}**."
        if positives
        else "All three pathology heads remain below their active decision thresholds."
    )
    triage_text = (
        "**Recommended action:** Route this exam to a radiologist for priority review."
        if data["triage_referred"]
        else "**Recommended action:** Low-uncertainty auto-screen candidate within the active budget."
    )
    prob_line = " | ".join(
        f"{LABEL_DISPLAY[lab]}: {probs[lab] * 100:.1f}% (thr {thresholds[lab] * 100:.1f}%)"
        for lab in LABELS
    )
    return (
        f"**Exam #{data['exam_id']} — {data['model_name']}** "
        f"(`{data['calibration_mode']}` calibration, `{data['threshold_mode']}` threshold)\n\n"
        f"- **Summary finding:** {finding_text}\n"
        f"- **Calibrated probabilities:** {prob_line}\n"
        f"- **Uncertainty telemetry:** Macro predictive entropy `{ent['macro']:.3f}` nats · "
        f"Cross-encoder std `{dis['macro']:.3f}` · Exam percentile `{data['triage_percentile']:.0f}%`\n"
        f"- {triage_text}\n\n"
        "_Phase 4 structured research prototype template (Phase 6 LLM narrative + Phase 5 Grad-CAM "
        "anatomical localization plug directly into this slot)._"
    )


def analyze_exam(
    bundle: Phase4Bundle,
    exam_id: int,
    model_key: str = "top3_ensemble",
    calibration_mode: str = "platt",
    threshold_mode: str = "val_youden",
    referral_rate: float = 0.15,
    uncertainty_metric: str = "predictive_entropy",
) -> ExamAnalysisResult:
    """Analyze one held-out MRNet test exam with calibration, thresholding, and triage."""
    if exam_id not in bundle.exam_ids:
        raise KeyError(f"Exam ID {exam_id} not found in held-out test split (1130..1249).")

    exam_idx = bundle.exam_ids.index(exam_id)
    test_df = bundle.test_dfs[model_key]
    raw_row = _df_prob_matrix(test_df)[exam_idx]
    gt_row = _df_label_matrix(test_df)[exam_idx]

    p_test_cal, pred_entropy, disagreement = _compute_uncertainty_matrices(
        bundle, model_key, calibration_mode
    )
    thresholds = get_thresholds(bundle, model_key, calibration_mode, threshold_mode)

    probs = {lab: float(p_test_cal[exam_idx, j]) for j, lab in enumerate(LABELS)}
    raw_probs = {lab: float(raw_row[j]) for j, lab in enumerate(LABELS)}
    predictions = {lab: int(probs[lab] >= thresholds[lab]) for lab in LABELS}
    ground_truth = {lab: int(gt_row[j]) for j, lab in enumerate(LABELS)}

    per_encoder_probs: dict[str, dict[str, float]] = {}
    for enc in BASE_ENCODERS:
        enc_cal = apply_calibration(bundle, enc, _df_prob_matrix(bundle.test_dfs[enc]), calibration_mode)
        per_encoder_probs[enc] = {lab: float(enc_cal[exam_idx, j]) for j, lab in enumerate(LABELS)}

    per_label_entropy = {lab: float(pred_entropy[exam_idx, j]) for j, lab in enumerate(LABELS)}
    per_label_entropy["macro"] = float(pred_entropy[exam_idx].mean())

    per_label_disagreement = {lab: float(disagreement[exam_idx, j]) for j, lab in enumerate(LABELS)}
    per_label_disagreement["macro"] = float(disagreement[exam_idx].mean())

    macro_scores = (
        disagreement.mean(axis=1)
        if uncertainty_metric == "encoder_disagreement"
        else pred_entropy.mean(axis=1)
    )
    referred, percentile, reason = _triage_status_for_exam(
        exam_idx=exam_idx,
        macro_scores=macro_scores,
        referral_rate=referral_rate,
        uncertainty_metric=uncertainty_metric,
        per_label_entropy=per_label_entropy,
        per_label_disagreement=per_label_disagreement,
    )

    gradcam = load_gradcam_overlays(exam_id)
    preliminary = {
        "exam_id": int(exam_id),
        "model_key": model_key,
        "model_name": MODEL_DISPLAY[model_key],
        "calibration_mode": calibration_mode,
        "threshold_mode": threshold_mode,
        "probs": probs,
        "raw_probs": raw_probs,
        "thresholds": thresholds,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "per_encoder_probs": per_encoder_probs,
        "predictive_entropy": per_label_entropy,
        "encoder_disagreement": per_label_disagreement,
        "triage_referred": referred,
        "triage_percentile": percentile,
        "triage_reason": reason,
        "gradcam_overlays": gradcam,
    }
    summary = generate_clinical_summary(preliminary)
    return ExamAnalysisResult(clinical_summary=summary, **preliminary)


def get_curated_exam_presets(bundle: Phase4Bundle) -> dict[str, int]:
    """Select 4 distinct illustrative test exam IDs (ACL, Healthy, Meniscus, Uncertain)."""
    test_df = bundle.test_dfs["top3_ensemble"]
    y = _df_label_matrix(test_df)
    p_platt = apply_calibration(bundle, "top3_ensemble", _df_prob_matrix(test_df), "platt")
    thr = get_thresholds(bundle, "top3_ensemble", "platt", "val_youden")
    macro_entropy = _binary_entropy(p_platt).mean(axis=1)
    used: set[int] = set()

    def _pick(mask: np.ndarray, score: np.ndarray, maximize: bool = False) -> int:
        candidates = [
            idx for idx in np.flatnonzero(mask) if bundle.exam_ids[int(idx)] not in used
        ]
        if not candidates:
            candidates = [i for i in range(len(bundle.exam_ids)) if bundle.exam_ids[i] not in used]
        chosen = max(candidates, key=lambda i: score[i]) if maximize else min(candidates, key=lambda i: score[i])
        eid = int(bundle.exam_ids[int(chosen)])
        used.add(eid)
        return eid

    uncertain_id = _pick(np.ones(len(bundle.exam_ids), dtype=bool), macro_entropy, maximize=True)
    acl_mask = (y[:, 1] == 1) & (p_platt[:, 1] >= max(0.75, thr["acl"])) & (p_platt[:, 0] >= 0.75)
    acl_id = _pick(acl_mask, macro_entropy, maximize=False)
    healthy_mask = (y[:, 0] == 0) & (y[:, 1] == 0) & (y[:, 2] == 0) & (p_platt[:, 0] <= 0.25)
    healthy_id = _pick(healthy_mask, macro_entropy, maximize=False)
    men_mask = (y[:, 2] == 1) & (y[:, 1] == 0) & (p_platt[:, 2] >= thr["meniscus"])
    men_score = p_platt[:, 2] - 0.35 * p_platt[:, 1]
    meniscus_id = _pick(men_mask, men_score, maximize=True)

    return {
        "Clear ACL Tear": acl_id,
        "Healthy / Normal Knee": healthy_id,
        "Meniscus Tear Case": meniscus_id,
        "Uncertain Case (Refer to Radiologist)": uncertain_id,
    }


def load_exam_slices(
    exam_id: int,
    data_root: Path = REPO_ROOT / "data" / "raw" / "MRNet-v1.0",
) -> dict[str, np.ndarray] | None:
    """Load local MRNet `valid` `.npy` slice volumes for `(axial, coronal, sagittal)` if present."""
    valid_dir = data_root / "valid"
    if not valid_dir.exists():
        return None

    volumes: dict[str, np.ndarray] = {}
    for plane in PLANES:
        vol_path = valid_dir / plane / f"{exam_id:04d}.npy"
        if not vol_path.exists():
            return None
        volumes[plane] = np.load(vol_path, mmap_mode="r")
    return volumes


def get_benchmark_dataframe(bundle: Phase4Bundle) -> pd.DataFrame:
    """Build the 6-model Val/Test benchmark summary DataFrame directly from `phase4_metrics.json`."""
    rows: list[dict[str, Any]] = []
    for key in ALL_MODELS:
        m = bundle.metrics["models"][key]
        va = m["val_auc"]
        ta = m["test_auc"]
        raw_cal = m["test_calibration"]["raw"]["macro"]
        platt_cal = m["test_calibration"]["platt"]["macro"]
        raw_f1 = float(
            np.mean([m["test_threshold_metrics"]["raw_0.50"][lab]["f1"] for lab in LABELS])
        )
        youden_f1 = float(
            np.mean([m["test_threshold_metrics"]["platt_val_youden"][lab]["f1"] for lab in LABELS])
        )
        rows.append(
            {
                "model_key": key,
                "Model": m["display_name"],
                "Val Macro AUC": round(float(va["macro_auc"]), 4),
                "Test Macro AUC": round(float(ta["macro_auc"]), 4),
                "Abnormal AUC (95% CI)": (
                    f"{ta['abnormal']['auc']:.4f} "
                    f"[{ta['abnormal']['ci95'][0]:.3f}–{ta['abnormal']['ci95'][1]:.3f}]"
                ),
                "ACL AUC (95% CI)": (
                    f"{ta['acl']['auc']:.4f} "
                    f"[{ta['acl']['ci95'][0]:.3f}–{ta['acl']['ci95'][1]:.3f}]"
                ),
                "Meniscus AUC (95% CI)": (
                    f"{ta['meniscus']['auc']:.4f} "
                    f"[{ta['meniscus']['ci95'][0]:.3f}–{ta['meniscus']['ci95'][1]:.3f}]"
                ),
                "Raw ECE": round(float(raw_cal["ece"]), 4),
                "Platt ECE": round(float(platt_cal["ece"]), 4),
                "Platt Brier": round(float(platt_cal["brier"]), 4),
                "Raw 0.50 Macro F1": round(raw_f1, 4),
                "Platt Youden Macro F1": round(youden_f1, 4),
            }
        )
    return pd.DataFrame(rows)


def get_delong_pair(
    bundle: Phase4Bundle,
    model_a: str,
    model_b: str,
    label: str,
) -> dict[str, Any]:
    """Look up paired DeLong ROC-AUC comparison in either key order."""
    if model_a == model_b:
        auc_same = float(bundle.metrics["models"][model_a]["test_auc"][label]["auc"])
        return {
            "model_a": model_a,
            "model_b": model_b,
            "label": label,
            "auc_a": auc_same,
            "auc_b": auc_same,
            "diff": 0.0,
            "se": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "significant_05": False,
        }

    pair_map = bundle.metrics["delong_test"][label]
    fwd = f"{model_a}__vs__{model_b}"
    rev = f"{model_b}__vs__{model_a}"
    if fwd in pair_map:
        raw = pair_map[fwd]
        return {
            "model_a": model_a,
            "model_b": model_b,
            "label": label,
            "auc_a": float(raw["auc_a"]),
            "auc_b": float(raw["auc_b"]),
            "diff": float(raw["diff"]),
            "se": float(raw["se"]),
            "z": float(raw["z"]),
            "p_value": float(raw["p_value"]),
            "significant_05": bool(raw["p_value"] < 0.05),
        }
    if rev in pair_map:
        raw = pair_map[rev]
        return {
            "model_a": model_a,
            "model_b": model_b,
            "label": label,
            "auc_a": float(raw["auc_b"]),
            "auc_b": float(raw["auc_a"]),
            "diff": float(-raw["diff"]),
            "se": float(raw["se"]),
            "z": float(-raw["z"]),
            "p_value": float(raw["p_value"]),
            "significant_05": bool(raw["p_value"] < 0.05),
        }
    raise KeyError(f"DeLong comparison not found for {model_a} vs {model_b} ({label}).")


def get_calibration_comparison(
    bundle: Phase4Bundle,
    model_key: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Return `(summary_df, prob_variants, y_test_vec)` for one model and pathology."""
    test_df = bundle.test_dfs[model_key]
    y_test = _df_label_matrix(test_df)
    p_raw = _df_prob_matrix(test_df)
    p_temp = apply_calibration(bundle, model_key, p_raw, "temperature")
    p_platt = apply_calibration(bundle, model_key, p_raw, "platt")
    variants = {"Raw": p_raw, "Temperature (Val)": p_temp, "Platt (Val)": p_platt}

    j = LABELS.index(label)
    rows: list[dict[str, Any]] = []
    for name, arr in variants.items():
        rep = calibration_report(y_test, arr, list(LABELS))
        lab_rep = rep["per_label"][label]
        macro_rep = rep["macro"]
        rows.append(
            {
                "Variant": name,
                f"{LABEL_DISPLAY[label]} ECE": round(float(lab_rep["ece"]), 4),
                f"{LABEL_DISPLAY[label]} Brier": round(float(lab_rep["brier"]), 4),
                f"{LABEL_DISPLAY[label]} NLL": round(float(lab_rep["nll"]), 4),
                "Macro ECE": round(float(macro_rep["ece"]), 4),
                "Macro Brier": round(float(macro_rep["brier"]), 4),
                "Macro NLL": round(float(macro_rep["nll"]), 4),
            }
        )
    return pd.DataFrame(rows), variants, y_test


def compute_selective_triage(
    bundle: Phase4Bundle,
    model_key: str = "top3_ensemble",
    calibration_mode: str = "platt",
    threshold_mode: str = "val_youden",
    uncertainty_metric: str = "predictive_entropy",
    referral_rate: float = 0.15,
) -> dict[str, Any]:
    """Compute per-label referral curves (`0%`..`30%`) and the flagged test exam table."""
    test_df = bundle.test_dfs[model_key]
    y_test = _df_label_matrix(test_df)
    p_cal, pred_entropy, disagreement = _compute_uncertainty_matrices(
        bundle, model_key, calibration_mode
    )
    thresholds = get_thresholds(bundle, model_key, calibration_mode, threshold_mode)
    u_matrix = disagreement if uncertainty_metric == "encoder_disagreement" else pred_entropy

    fractions = list(REFERRAL_FRACTIONS)
    if round(float(referral_rate), 4) not in {round(f, 4) for f in fractions}:
        fractions = sorted(set(fractions + [float(referral_rate)]))

    curves: dict[str, list[dict[str, float]]] = {}
    selected_rows: dict[str, dict[str, float]] = {}
    baseline_rows: dict[str, dict[str, float]] = {}
    for j, lab in enumerate(LABELS):
        curve = selective_referral_curve(
            y_test[:, j],
            p_cal[:, j],
            u_matrix[:, j],
            fractions=fractions,
            threshold=thresholds[lab],
        )
        curves[lab] = curve
        baseline_rows[lab] = curve[0]
        selected_rows[lab] = min(curve, key=lambda r: abs(r["referral_fraction"] - referral_rate))

    macro_u = u_matrix.mean(axis=1)
    n_total = len(bundle.exam_ids)
    n_refer = int(round(float(referral_rate) * n_total))
    desc_idx = np.argsort(-macro_u, kind="mergesort")[:n_refer] if n_refer > 0 else np.array([], dtype=int)

    flagged_records: list[dict[str, Any]] = []
    for rank, idx in enumerate(desc_idx, start=1):
        rec: dict[str, Any] = {
            "Rank": rank,
            "Exam ID": int(bundle.exam_ids[int(idx)]),
            "Macro Uncertainty": round(float(macro_u[idx]), 4),
        }
        for j, lab in enumerate(LABELS):
            short = LABEL_DISPLAY[lab]
            rec[f"{short} Prob"] = round(float(p_cal[idx, j]), 3)
            rec[f"{short} Call"] = int(p_cal[idx, j] >= thresholds[lab])
            rec[f"{short} GT"] = int(y_test[idx, j])
        flagged_records.append(rec)

    return {
        "thresholds": thresholds,
        "curves": curves,
        "baseline": baseline_rows,
        "selected": selected_rows,
        "referred_count": n_refer,
        "kept_count": n_total - n_refer,
        "flagged_df": pd.DataFrame(flagged_records),
    }


def get_threshold_comparison_row(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str,
    threshold_mode: str,
) -> list[dict[str, Any]]:
    """Compute held-out Test classification metrics for the chosen calibration + threshold."""
    test_df = bundle.test_dfs[model_key]
    y_test = _df_label_matrix(test_df)
    p_cal = apply_calibration(bundle, model_key, _df_prob_matrix(test_df), calibration_mode)
    thresholds = get_thresholds(bundle, model_key, calibration_mode, threshold_mode)
    rows: list[dict[str, Any]] = []
    for j, lab in enumerate(LABELS):
        m = threshold_metrics(y_test[:, j], p_cal[:, j], thresholds[lab])
        rows.append(
            {
                "Pathology": LABEL_DISPLAY[lab],
                "Threshold": round(float(m["threshold"]), 3),
                "Accuracy": round(float(m["accuracy"]), 4),
                "Sensitivity": round(float(m["sensitivity"]), 4),
                "Specificity": round(float(m["specificity"]), 4),
                "Precision": round(float(m["precision"]), 4),
                "F1": round(float(m["f1"]), 4),
            }
        )
    return rows
