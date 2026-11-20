# OrthoVision AI — Streamlit Interactive Demo (Phase 4 Ready)

> **Research prototype only — not for clinical diagnosis.**

This directory holds the interactive demo dashboard built on locked Phase 3/4 outputs (`outputs/logs/phase4_metrics.json`, `outputs/predictions/*_seed42_{val,test}.csv`, and `outputs/logs/*_seed42/log.csv`), architected so Phases 5 and 6 drop in cleanly without rewriting the UI.

## How to Run Locally (CPU Only — No PyTorch/GPU Needed)

From the repository root (`orthovision/`):

```bash
streamlit run app/app.py
```

### Prerequisites & Data Bundles
- **Python packages:** `streamlit`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `Pillow`, `pyyaml`.
- **Public Bundle (`phase4_outputs_public.zip`, `~53 KB`):** unzip into the repo root (`orthovision/`) to populate git-ignored `outputs/predictions/` and `outputs/logs/` (zero MRI images; safe for public sharing).
- **Knee MRI Scan Display (optional, auto-detected):**
  - **Private Team Bundle (`mrnet_test_previews_private.zip`, `72.83 MB`):** team-only archive on Google Drive (`OrthoVision Backups / Phase4_Demo_Pack / Private_Team_Only_MRNet_Previews`). Unzip into `orthovision/` to populate git-ignored `data/processed/exam_previews/{axial,coronal,sagittal}/` (`1130_00.png`..`1249_04.png`).
  - **Full MRNet `.npy` volumes (if you hold Stanford MRNet access):** place scans under `data/raw/MRNet-v1.0/valid/` (or point `ORTHOVISION_DATA_ROOT` / `configs/local.yaml` -> `paths.data_root` to your local MRNet folder).
  When present, the **Exam Explorer** shows interactive Axial / Coronal / Sagittal MRI slices; when absent, it degrades gracefully to placeholder cards while all 4 analytics tabs remain fully interactive.

## How Phases 5 & 6 Plug Into `app/demo_backend.py`
- **Phase 5 (Grad-CAM + Region Mapping):** implement `load_gradcam_overlays(exam_id)` to return per-plane overlay images and anatomical region scores inside `ExamAnalysisResult.gradcam_overlays`.
- **Phase 6 (LLM Clinical Summary):** replace the structured template inside `generate_clinical_summary(result_dict)` with the grounded LLM narrative generator.
