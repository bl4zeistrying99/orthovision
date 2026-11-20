"""OrthoVision AI — Phase 4 Interactive Research Demo (Streamlit Dashboard).

Built on locked Phase 3/4 outputs (held-out MRNet test split, n = 120, Seed 42).
Architected cleanly around `app/demo_backend.py` so Phase 5 (Grad-CAM heatmaps &
region mapping) and Phase 6 (LLM clinical summary) plug in without rewriting the UI.

Run locally from the repository root:
    streamlit run app/app.py

Research prototype only — not for clinical diagnosis.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demo_backend import (  # noqa: E402
    ALL_MODELS,
    BASE_ENCODERS,
    LABEL_DISPLAY,
    LABELS,
    MODEL_DISPLAY,
    MODEL_SHORT,
    PLANES,
    MissingOutputsError,
    Phase4Bundle,
    analyze_exam,
    apply_calibration,
    compute_selective_triage,
    get_benchmark_dataframe,
    get_calibration_comparison,
    get_curated_exam_presets,
    get_delong_pair,
    get_threshold_comparison_row,
    load_exam_slices,
    load_phase4_bundle,
)
from orthovision.eval.calibration import expected_calibration_error  # noqa: E402

st.set_page_config(
    page_title="OrthoVision AI — Phase 4 Demo",
    page_icon="🦴",
    layout="wide",
)

MODEL_COLORS: dict[str, str] = {
    "top3_ensemble": "#0f766e",
    "resnet50": "#2563eb",
    "swin": "#7c3aed",
    "efficientnet_b0": "#d97706",
    "effnet_vit": "#0891b2",
    "vit": "#dc2626",
}
LABEL_COLORS: dict[str, str] = {
    "abnormal": "#2563eb",
    "acl": "#dc2626",
    "meniscus": "#0f766e",
}


@st.cache_resource
def _cached_bundle() -> Phase4Bundle:
    return load_phase4_bundle(repo_root=REPO_ROOT, seed=42)


def _plot_exam_encoder_agreement(result) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y_pos = np.arange(len(LABELS))
    series_keys = [*BASE_ENCODERS]
    if result.model_key not in BASE_ENCODERS:
        series_keys.append(result.model_key)

    n_bars = len(series_keys)
    height = 0.72 / n_bars
    offsets = np.linspace(-0.36 + height / 2, 0.36 - height / 2, n_bars)

    for idx, mkey in enumerate(series_keys):
        if mkey == result.model_key and mkey not in BASE_ENCODERS:
            vals = [result.probs[lab] for lab in LABELS]
            label_txt = f"{MODEL_SHORT[mkey]} (Active)"
        else:
            vals = [result.per_encoder_probs[mkey][lab] for lab in LABELS]
            label_txt = MODEL_SHORT[mkey]
        bars = ax.barh(
            y_pos + offsets[idx],
            vals,
            height=height * 0.92,
            color=MODEL_COLORS[mkey],
            alpha=0.92 if mkey == result.model_key else 0.72,
            label=label_txt,
        )
        if mkey == result.model_key:
            for bar in bars:
                ax.text(
                    min(bar.get_width() + 0.015, 0.98),
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.2f}",
                    va="center",
                    ha="left",
                    fontsize=8.0,
                    fontweight="bold",
                )

    for j, lab in enumerate(LABELS):
        thr = result.thresholds[lab]
        ax.plot([thr, thr], [j - 0.42, j + 0.42], "k--", lw=1.2, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABEL_DISPLAY[lab] for lab in LABELS], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("Calibrated Probability (dashed line = active threshold)")
    ax.set_title(f"Exam #{result.exam_id}: Cross-Encoder Agreement", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.0, ncol=2)
    fig.tight_layout()
    return fig


def _plot_roc_curves(
    bundle: Phase4Bundle,
    selected_models: list[str],
    pathology: str,
    calibration_mode: str,
) -> plt.Figure:
    j = LABELS.index(pathology)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for mkey in selected_models:
        df = bundle.test_dfs[mkey]
        y_true = df[f"y_{pathology}"].to_numpy(dtype=np.int64)
        p_raw = df[[f"p_{lab}" for lab in LABELS]].to_numpy(dtype=np.float64)
        p_cal = apply_calibration(bundle, mkey, p_raw, calibration_mode)[:, j]
        fpr, tpr, _ = roc_curve(y_true, p_cal)
        auc_val = roc_auc_score(y_true, p_cal)
        ax.plot(
            fpr,
            tpr,
            lw=2.3 if mkey == "top3_ensemble" else 1.8,
            color=MODEL_COLORS[mkey],
            label=f"{MODEL_SHORT[mkey]} (AUC = {auc_val:.4f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Chance (0.5000)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(f"Held-Out Test ROC Curves (n = 120) — {LABEL_DISPLAY[pathology]}", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    return fig


def _plot_training_curves(bundle: Phase4Bundle) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for enc in BASE_ENCODERS:
        log_df = bundle.train_logs[enc]
        col = MODEL_COLORS[enc]
        name = MODEL_SHORT[enc]
        best_idx = int(log_df["auc_macro"].idxmax())
        best_ep = int(log_df.loc[best_idx, "epoch"])
        best_auc = float(log_df.loc[best_idx, "auc_macro"])

        axes[0].plot(log_df["epoch"], log_df["auc_macro"], lw=2.0, color=col, label=f"{name} (ep {best_ep}: {best_auc:.3f})")
        axes[0].scatter([best_ep], [best_auc], color=col, s=55, zorder=5, edgecolor="white", linewidth=1.0)
        axes[1].plot(log_df["epoch"], log_df["train_loss"], lw=2.0, color=col, label=name)

    axes[0].set_title("Validation Macro AUC vs. Epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val Macro ROC-AUC (n = 170)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8.2)

    axes[1].set_title("Training Loss vs. Epoch (Kaggle T4)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted BCE Loss")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=8.2)

    fig.tight_layout()
    return fig


def _plot_reliability_single(
    y_test: np.ndarray,
    variants: dict[str, np.ndarray],
    model_name: str,
    pathology: str,
) -> plt.Figure:
    j = LABELS.index(pathology)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Perfectly calibrated")
    palette = ["#dc2626", "#f59e0b", "#0f766e"]
    markers = ["o", "s", "^"]
    for idx, (vname, probs) in enumerate(variants.items()):
        info = expected_calibration_error(y_test[:, j], probs[:, j], n_bins=10)
        xs = [c for c in info["bin_conf"] if c is not None]
        ys = [a for a in info["bin_acc"] if a is not None]
        ax.plot(
            xs,
            ys,
            marker=markers[idx],
            color=palette[idx],
            lw=2.1,
            ms=6.0,
            label=f"{vname} (ECE = {info['ece']:.4f})",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Fraction")
    ax.set_title(f"{model_name} Reliability Diagram — {LABEL_DISPLAY[pathology]}", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.8)
    fig.tight_layout()
    return fig


def _plot_selective_referral(triage_data: dict, active_pct: int, model_name: str, signal_name: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    curves = triage_data["curves"]
    for idx, lab in enumerate(LABELS):
        ax = axes[idx]
        rows = curves[lab]
        cov = [r["referral_fraction"] * 100.0 for r in rows]
        acc = [r["accuracy"] * 100.0 for r in rows]
        brier = [r["brier"] for r in rows]

        ax.plot(cov, acc, marker="o", lw=2.2, color=LABEL_COLORS[lab], label="Retained Accuracy (%)")
        ax.axvline(active_pct, color="#334155", linestyle="--", lw=1.4, label=f"Active ({active_pct}%)")
        ax.set_title(f"{LABEL_DISPLAY[lab]}", fontweight="bold")
        ax.set_xlabel("Uncertain Exams Referred (%)")
        ax.set_ylabel("Retained Accuracy (%)", color=LABEL_COLORS[lab])
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.grid(True, linestyle=":", alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(cov, brier, marker="s", linestyle=":", lw=1.8, color="#64748b", label="Brier Score")
        ax2.set_ylabel("Brier Score (lower better)", color="#64748b")

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=7.6)

    fig.suptitle(f"{model_name} — Selective Triage ({signal_name}, Test n = 120)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def _render_exam_explorer(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str,
    threshold_mode: str,
    uncertainty_metric: str,
    referral_rate: float,
) -> None:
    presets = get_curated_exam_presets(bundle)
    preset_options = [*presets.keys(), "Custom Test Exam ID (1130–1249)"]

    col_sel, col_id, col_gt = st.columns([2.2, 1.3, 1.0])
    with col_sel:
        selected_preset = st.selectbox(
            "Case Preset (Quick Walkthrough)",
            options=preset_options,
            index=0,
        )
    with col_id:
        if selected_preset in presets:
            exam_id = presets[selected_preset]
            st.selectbox(
                "MRNet Test Exam ID",
                options=list(bundle.exam_ids),
                index=bundle.exam_ids.index(exam_id),
                disabled=True,
                help="Switch preset to 'Custom Test Exam ID' to browse all 120 held-out exams.",
            )
        else:
            exam_id = st.selectbox(
                "MRNet Test Exam ID",
                options=list(bundle.exam_ids),
                index=0,
            )
    with col_gt:
        st.write("")
        reveal_gt = st.toggle("Reveal MRNet Ground Truth", value=True)

    result = analyze_exam(
        bundle=bundle,
        exam_id=exam_id,
        model_key=model_key,
        calibration_mode=calibration_mode,
        threshold_mode=threshold_mode,
        referral_rate=referral_rate,
        uncertainty_metric=uncertainty_metric,
    )

    if result.triage_referred:
        st.warning(
            f"**Triage Status: 🟠 Refer to Radiologist** (Uncertainty Percentile: "
            f"{result.triage_percentile:.0f}th) — {result.triage_reason}"
        )
    else:
        st.success(
            f"**Triage Status: 🟢 Auto-Screen Candidate** (Uncertainty Percentile: "
            f"{result.triage_percentile:.0f}th) — {result.triage_reason}"
        )

    c1, c2, c3 = st.columns(3)
    for col, lab in zip((c1, c2, c3), LABELS):
        with col:
            prob = result.probs[lab]
            raw_p = result.raw_probs[lab]
            thr = result.thresholds[lab]
            pred = result.predictions[lab]
            gt = result.ground_truth[lab]
            call_str = "POSITIVE" if pred == 1 else "NEGATIVE"
            st.metric(
                label=LABEL_DISPLAY[lab],
                value=f"{prob * 100:.1f}%",
                delta=f"{call_str} (thr {thr * 100:.1f}%)",
                delta_color="inverse" if pred == 1 else "normal",
            )
            st.progress(float(np.clip(prob, 0.0, 1.0)))
            caption_parts = [f"Raw prob: `{raw_p * 100:.1f}%`", f"Entropy: `{result.predictive_entropy[lab]:.3f}`"]
            if reveal_gt:
                gt_str = "Positive" if gt == 1 else "Negative"
                match_str = "Match ✅" if pred == gt else "Mismatch ❌"
                caption_parts.append(f"Ground Truth: **{gt_str}** ({match_str})")
            st.caption(" · ".join(caption_parts))

    st.divider()
    st.subheader(f"Knee MRI Scan Viewer — Exam #{exam_id} (Axial · Coronal · Sagittal)")
    slice_bundle = load_exam_slices(exam_id)
    if slice_bundle is not None:
        planes_map = slice_bundle["planes"]
        source_tag = slice_bundle["source"]
        img_cols = st.columns(3)
        slice_counts = []
        for p_col, plane in zip(img_cols, PLANES):
            vol = planes_map[plane]
            n_slices = int(vol.shape[0])
            slice_counts.append(n_slices)
            with p_col:
                st.markdown(f"**{plane.capitalize()} View**")
                if n_slices > 1:
                    s_idx = st.slider(
                        f"{plane.capitalize()} slice",
                        min_value=0,
                        max_value=n_slices - 1,
                        value=n_slices // 2,
                        key=f"slice_{exam_id}_{plane}",
                    )
                else:
                    s_idx = 0
                st.image(
                    vol[s_idx],
                    caption=f"{plane.capitalize()} · Slice {s_idx + 1} / {n_slices}",
                    use_container_width=True,
                    clamp=True,
                )
        if source_tag == "mrnet_npy":
            st.caption(
                "Source: Full MRNet `.npy` volume (`data/raw/MRNet-v1.0/valid/` or configured override) · "
                "Phase 5 Grad-CAM heatmaps will overlay directly onto these 3 planes."
            )
        else:
            max_k = max(slice_counts) if slice_counts else 1
            st.caption(
                f"Source: {max_k}-slice MRNet PNG preview (`data/processed/exam_previews/`) · "
                "Phase 5 Grad-CAM heatmaps will overlay directly onto these 3 planes."
            )
    else:
        plane_cards = st.columns(3)
        for p_col, plane, focus in zip(
            plane_cards,
            ("Axial", "Coronal", "Sagittal"),
            ("Joint effusion & patellar tracking", "Meniscal body & collateral view", "ACL course & posterior horns"),
        ):
            with p_col:
                st.markdown(
                    f"<div style='border:1px dashed #94a3b8; border-radius:8px; padding:14px; text-align:center;'>"
                    f"<strong>{plane} View</strong><br><span style='font-size:12px; color:#64748b;'>{focus}</span><br>"
                    f"<code style='font-size:11px;'>MRI image slot ready</code></div>",
                    unsafe_allow_html=True,
                )
        st.caption(
            "No local MRI image files found yet in `data/processed/exam_previews/` or `data/raw/MRNet-v1.0/valid/`. "
            "Drop in exported PNG previews or full MRNet `.npy` scans and refresh — photos appear here automatically."
        )

    st.divider()
    left_col, right_col = st.columns([1.15, 1.0])
    with left_col:
        st.subheader("Cross-Encoder Agreement")
        fig_agree = _plot_exam_encoder_agreement(result)
        st.pyplot(fig_agree, use_container_width=True)
        plt.close(fig_agree)

        st.subheader("Research Summary Card (Phase 6 LLM Hook)")
        st.info(result.clinical_summary)

    with right_col:
        st.subheader("Per-Encoder Breakdown & Phase 5 Hook")
        enc_rows = [
            {
                "Encoder": MODEL_SHORT[enc],
                "Abnormal": round(result.per_encoder_probs[enc]["abnormal"], 3),
                "ACL Tear": round(result.per_encoder_probs[enc]["acl"], 3),
                "Meniscus Tear": round(result.per_encoder_probs[enc]["meniscus"], 3),
            }
            for enc in BASE_ENCODERS
        ]
        st.markdown("**Per-Encoder Calibrated Probabilities**")
        st.dataframe(pd.DataFrame(enc_rows), use_container_width=True, hide_index=True)

        st.info(
            "**Phase 5 Grad-CAM Slot Ready:** `ExamAnalysisResult.gradcam_overlays` and "
            "`load_gradcam_overlays()` in `app/demo_backend.py` are already wired to overlay attention "
            "heatmaps & anatomical region mapping directly onto the 3 MRI views above."
        )


def _render_benchmark_tab(bundle: Phase4Bundle, calibration_mode: str) -> None:
    st.subheader("Locked Identical-Protocol Benchmark (Held-Out MRNet Test, n = 120)")
    top3 = bundle.metrics["models"]["top3_ensemble"]
    r50 = bundle.metrics["models"]["resnet50"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Test Macro AUC", f"{top3['test_auc']['macro_auc']:.4f}", "Top-3 Ensemble")
    k2.metric("Best ACL Tear AUC", f"{top3['test_auc']['acl']['auc']:.4f}", "Top-3 Ensemble")
    k3.metric("Best Single Encoder", f"{r50['test_auc']['macro_auc']:.4f}", "ResNet50 (Ep 16)")
    k4.metric(
        "Lowest Raw ECE",
        f"{top3['test_calibration']['raw']['macro']['ece']:.4f}",
        "47% lower than ResNet50",
    )

    bench_df = get_benchmark_dataframe(bundle).drop(columns=["model_key"])
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    c_roc, c_trn = st.columns(2)
    with c_roc:
        st.markdown("**Interactive Test ROC Curves**")
        sel_models = st.multiselect(
            "Compare Models",
            options=list(ALL_MODELS),
            default=["top3_ensemble", "resnet50", "swin", "efficientnet_b0", "vit"],
            format_func=lambda k: MODEL_SHORT[k],
        )
        roc_label = st.radio(
            "ROC Pathology",
            options=list(LABELS),
            format_func=lambda k: LABEL_DISPLAY[k],
            horizontal=True,
        )
        if sel_models:
            fig_roc = _plot_roc_curves(bundle, sel_models, roc_label, calibration_mode)
            st.pyplot(fig_roc, use_container_width=True)
            plt.close(fig_roc)
        else:
            st.info("Select at least one model to plot ROC curves.")

    with c_trn:
        st.markdown("**Kaggle Training Progression Across All 4 Encoders**")
        fig_trn = _plot_training_curves(bundle)
        st.pyplot(fig_trn, use_container_width=True)
        plt.close(fig_trn)
        st.caption(
            "Stars mark the checkpoint epoch with highest Validation Macro AUC (`resnet50` Ep 16, "
            "`efficientnet_b0` Ep 9, `swin` Ep 18, `vit` Ep 14). All 4 encoders trained under identical `configs/default.yaml`."
        )


def _render_calibration_delong_tab(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str,
    threshold_mode: str,
) -> None:
    st.subheader(f"Calibration & Threshold Analysis — {MODEL_DISPLAY[model_key]}")
    cal_label = st.radio(
        "Inspect Pathology Calibration",
        options=list(LABELS),
        format_func=lambda k: LABEL_DISPLAY[k],
        horizontal=True,
        key="cal_pathology_radio",
    )
    cal_df, variants, y_test = get_calibration_comparison(bundle, model_key, cal_label)

    c_left, c_right = st.columns([1.05, 1.0])
    with c_left:
        fig_rel = _plot_reliability_single(y_test, variants, MODEL_SHORT[model_key], cal_label)
        st.pyplot(fig_rel, use_container_width=True)
        plt.close(fig_rel)
    with c_right:
        st.markdown("**Calibration Metrics (Test Split, n = 120; Fitted on Val Only)**")
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

        st.markdown(f"**Classification Metrics (`{calibration_mode}` · `{threshold_mode}`)**")
        thr_rows = get_threshold_comparison_row(bundle, model_key, calibration_mode, threshold_mode)
        st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Note: Weighted BCE (`use_pos_weight: true`) shifts raw logits away from dataset prevalence; "
            "Validation-only Platt Scaling + Youden thresholding improves operating F1 without touching the test set."
        )

    st.divider()
    st.subheader("Paired DeLong ROC-AUC Significance Test (Held-Out Test, n = 120)")
    d1, d2, d3 = st.columns(3)
    with d1:
        model_a = st.selectbox(
            "Model A",
            options=list(ALL_MODELS),
            index=0,
            format_func=lambda k: MODEL_SHORT[k],
            key="delong_a",
        )
    with d2:
        model_b = st.selectbox(
            "Model B",
            options=list(ALL_MODELS),
            index=5,
            format_func=lambda k: MODEL_SHORT[k],
            key="delong_b",
        )
    with d3:
        delong_label = st.selectbox(
            "Pathology",
            options=list(LABELS),
            index=1,
            format_func=lambda k: LABEL_DISPLAY[k],
            key="delong_lab",
        )

    res = get_delong_pair(bundle, model_a, model_b, delong_label)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{MODEL_SHORT[model_a]} AUC", f"{res['auc_a']:.4f}")
    m2.metric(f"{MODEL_SHORT[model_b]} AUC", f"{res['auc_b']:.4f}")
    m3.metric("ΔAUC (A - B)", f"{res['diff']:+.4f}", f"z = {res['z']:+.3f}")
    m4.metric("Two-Sided p-value", f"{res['p_value']:.4f}", "p < 0.05 ✅" if res["significant_05"] else "Not significant")

    if res["significant_05"]:
        st.success(
            f"**Statistically significant difference (p = {res['p_value']:.4f} < 0.05):** "
            f"{MODEL_SHORT[model_a]} and {MODEL_SHORT[model_b]} differ significantly on "
            f"{LABEL_DISPLAY[delong_label]} ROC-AUC (`ΔAUC = {res['diff']:+.4f}`)."
        )
    else:
        st.info(
            f"**No statistically significant difference at α = 0.05 (p = {res['p_value']:.4f}):** "
            f"At n = 120 test exams, the observed gap (`ΔAUC = {res['diff']:+.4f}`) could arise from sample variance."
        )


def _render_triage_tab(
    bundle: Phase4Bundle,
    model_key: str,
    calibration_mode: str,
    threshold_mode: str,
    uncertainty_metric: str,
    referral_rate: float,
    referral_pct: int,
) -> None:
    triage = compute_selective_triage(
        bundle=bundle,
        model_key=model_key,
        calibration_mode=calibration_mode,
        threshold_mode=threshold_mode,
        uncertainty_metric=uncertainty_metric,
        referral_rate=referral_rate,
    )
    signal_name = (
        "4-Encoder Disagreement (Std)"
        if uncertainty_metric == "encoder_disagreement"
        else "Predictive Entropy"
    )
    st.subheader(
        f"Selective Radiologist Triage — {MODEL_SHORT[model_key]} ({signal_name}, Refer Top {referral_pct}%)"
    )

    cols = st.columns(3)
    for col, lab in zip(cols, LABELS):
        base = triage["baseline"][lab]
        kept = triage["selected"][lab]
        delta_acc = (kept["accuracy"] - base["accuracy"]) * 100.0
        with col:
            st.metric(
                label=f"{LABEL_DISPLAY[lab]} Retained Accuracy ({triage['kept_count']}/120)",
                value=f"{kept['accuracy'] * 100:.1f}%",
                delta=f"{delta_acc:+.1f}% vs all 120 ({base['accuracy'] * 100:.1f}%)",
            )
            auc_str = f"{kept['roc_auc']:.4f}" if np.isfinite(kept["roc_auc"]) else "n/a"
            st.caption(
                f"Retained AUC: `{auc_str}` (base `{base['roc_auc']:.4f}`) · "
                f"Brier: `{kept['brier']:.4f}` (base `{base['brier']:.4f}`)"
            )

    fig_ref = _plot_selective_referral(triage, referral_pct, MODEL_SHORT[model_key], signal_name)
    st.pyplot(fig_ref, use_container_width=True)
    plt.close(fig_ref)

    st.markdown(f"**Exams Flagged for Radiologist Review ({triage['referred_count']} of 120 Test Exams)**")
    if triage["referred_count"] == 0:
        st.info("Referral budget is currently 0%. Move the sidebar slider above 0% to inspect the most uncertain exams.")
    else:
        st.dataframe(triage["flagged_df"], use_container_width=True, hide_index=True)


def main() -> None:
    st.title("OrthoVision AI — Knee MRI Decision Support (Phase 4 Interactive Demo)")
    st.warning(
        "**Research Prototype Only — Not for Clinical Diagnosis.** "
        "Running live on locked Phase 4 held-out MRNet test outputs (`n = 120`, Seed `42`). "
        "Phase 5 (Grad-CAM heatmaps & anatomical region mapping) and Phase 6 (LLM clinical summary) "
        "plug directly into the modular hooks in `app/demo_backend.py`."
    )

    try:
        bundle = _cached_bundle()
    except MissingOutputsError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Analysis Controls")
        model_key = st.selectbox(
            "Model / Ensemble",
            options=list(ALL_MODELS),
            index=0,
            format_func=lambda k: MODEL_DISPLAY[k],
        )
        calibration_mode = st.radio(
            "Probability Calibration (Val-Fit)",
            options=["platt", "temperature", "raw"],
            index=0,
            format_func={
                "platt": "Platt Scaling (Val-Fit)",
                "temperature": "Temperature Scaling (Val-Fit)",
                "raw": "Raw Uncalibrated",
            }.__getitem__,
        )
        threshold_mode = st.radio(
            "Decision Threshold",
            options=["val_youden", "fixed_0.50"],
            index=0,
            format_func={
                "val_youden": "Val-Tuned Youden's J",
                "fixed_0.50": "Fixed 0.50",
            }.__getitem__,
        )
        uncertainty_metric = st.radio(
            "Triage Uncertainty Signal",
            options=["predictive_entropy", "encoder_disagreement"],
            index=0,
            format_func={
                "predictive_entropy": "Predictive Entropy",
                "encoder_disagreement": "4-Encoder Disagreement (Std)",
            }.__getitem__,
        )
        referral_pct = st.slider(
            "Radiologist Referral Budget (%)",
            min_value=0,
            max_value=30,
            value=15,
            step=5,
            format="%d%%",
        )
        referral_rate = referral_pct / 100.0

        st.divider()
        st.markdown(
            """
            **Pipeline Status**
            - ✅ Phase 0–3: Frozen split & 4 Kaggle encoders
            - ✅ Phase 4: Calibration, DeLong & Triage
            - ⏳ Phase 5: Grad-CAM + Region Mapping *(hook ready)*
            - ⏳ Phase 6: LLM Narrative Layer *(hook ready)*
            """
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 Exam Explorer",
            "📊 4-Encoder & Ensemble Benchmark",
            "🎯 Calibration & DeLong Tests",
            "🩺 Uncertainty-Aware Triage",
        ]
    )
    with tab1:
        _render_exam_explorer(
            bundle,
            model_key,
            calibration_mode,
            threshold_mode,
            uncertainty_metric,
            referral_rate,
        )
    with tab2:
        _render_benchmark_tab(bundle, calibration_mode)
    with tab3:
        _render_calibration_delong_tab(bundle, model_key, calibration_mode, threshold_mode)
    with tab4:
        _render_triage_tab(
            bundle,
            model_key,
            calibration_mode,
            threshold_mode,
            uncertainty_metric,
            referral_rate,
            referral_pct,
        )


if __name__ == "__main__":
    main()
