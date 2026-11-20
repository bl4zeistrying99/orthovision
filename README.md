# OrthoVision AI

**Explainable and Uncertainty-Aware Knee MRI Analysis and Clinical Decision Support System**

> **Research Prototype Only — Not for Clinical Diagnosis.**  
> B.Tech CSE Final-Year Major Project · Graphic Era Deemed to be University, Dehradun · **Team ID:** `MP2026CSE223`  
> **Guide:** Mr. Vishal Trivedi · **Team:** Bhawesh Gunjiyal, Rudra Pratap Singh, Amey Jhaldiyal, Daksh Choudhary

---

## Project Progress at a Glance

| Phase | Scope | Status |
|---|---|---|
| **Phase 0** | Repo scaffold, identical-protocol config (`configs/default.yaml`), dataset loader, unit tests | Done |
| **Phase 1** | MRNet data verification, EDA, frozen exam-level split (`splits/split_v1.json`) | Done |
| **Phase 2** | Shared multi-plane `ExamModel` (`timm` + max-pool + 3 heads + dropout), GPU training engine, ResNet50 baseline | Done |
| **Phase 3** | All 4 encoders trained on Kaggle Tesla T4 under identical protocol (`ResNet50`, `EfficientNet-B0`, `Swin-Tiny`, `ViT-Base`) | Done |
| **Phase 4** | Ensembles (`Top-3` & `EffNet+ViT`), Val-fitted calibration (Temperature & Platt), Val Youden thresholds, DeLong tests, Selective Triage, **Interactive Streamlit Demo** | Done |
| **Phase 5** | Grad-CAM visual heatmaps & anatomical region mapping (`src/orthovision/explain/`) | Upcoming |
| **Phase 6** | Grounded LLM clinical narrative layer (`src/orthovision/llm/`) + final ablations & report | Upcoming |

New to the repo? Read the plain-English teammate & commit guide in [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md).

---

## Locked Benchmark Results (Held-Out MRNet Test Split, `n = 120`, Seed `42`)

All 4 base encoders were trained on Kaggle Tesla T4 with identical hyperparameters (`configs/default.yaml`). Probability calibration (Temperature & Platt scaling) and Youden operating thresholds were fitted **strictly on Validation (`n = 170`)** — never on the Test set.

| Model / Ensemble | Val Macro AUC | Test Macro AUC | Abnormal AUC | ACL AUC | Meniscus AUC | Raw ECE | Platt ECE | Platt Youden Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Top-3 Ensemble (`R50 + EffNet-B0 + Swin`)** | 0.8815 | **0.9000** | 0.9516 | **0.9632** | **0.7851** | **0.0563** | 0.0772 | **0.8408** |
| **ResNet50** (Best Ep 16) | **0.8893** | 0.8944 | **0.9571** | 0.9610 | 0.7653 | 0.1056 | 0.0933 | 0.8283 |
| **Swin-Tiny** (Best Ep 18) | 0.8549 | 0.8756 | 0.9406 | 0.9198 | 0.7664 | 0.1203 | 0.1054 | 0.8078 |
| **EfficientNet-B0** (Best Ep 9) | 0.8509 | 0.8716 | 0.9183 | 0.9287 | 0.7677 | 0.1059 | 0.1023 | 0.8154 |
| **EffNet-B0 + ViT** (Synopsis Pair) | 0.8349 | 0.8684 | 0.9112 | 0.9122 | 0.7820 | 0.0899 | **0.0864** | 0.8184 |
| **ViT-Base** (Best Ep 14) | 0.8037 | 0.8158 | 0.8684 | 0.8363 | 0.7428 | 0.0888 | 0.1061 | 0.7281 |

### Key Takeaways from Phase 4
1. **Ensemble beats every individual backbone:** `Top-3 Ensemble` reaches **`0.9000` Test Macro AUC**, highest **ACL (`0.9632`)** and **Meniscus (`0.7851`)** AUCs, and cuts raw calibration error nearly in half (`ECE 0.1056 -> 0.0563`, `Brier 0.1258 -> 0.1144`).
2. **Statistical significance (Paired DeLong on Test `n = 120`):** On ACL Tear, `Top-3 Ensemble` significantly outperforms `EfficientNet-B0` (`p = 0.0315`), `Swin-Tiny` (`p = 0.0357`), `EffNet+ViT` (`p = 0.0128`), and `ViT-Base` (`p = 0.0002`). `EffNet-B0 + ViT` also beats standalone `ViT-Base` on both ACL (`p = 0.0006`) and Abnormal (`p = 0.0217`).
3. **Why calibration + Youden thresholding matter:** Training with `use_pos_weight: true` boosts recall under class imbalance, shifting raw probabilities away from true prevalence. Validation-only Platt scaling + Youden thresholding improves Macro F1 across top models (`Swin-Tiny: 0.7564 -> 0.8078`, `Top-3 Ensemble: 0.8299 -> 0.8408`).
4. **Uncertainty-aware radiologist referral works:** Referring the top 25% most uncertain exams (by Top-3 predictive entropy, 30/120 cases) boosts accuracy on the remaining 90 exams:
   - **Abnormal:** `85.0% -> 95.6%` (Brier `0.0901 -> 0.0389`)
   - **ACL Tear:** `88.3% -> 96.7%` (ROC-AUC `0.9632 -> 0.9905`, Brier `0.0872 -> 0.0357`)
   - **Meniscus Tear:** `71.7% -> 75.6%` (ROC-AUC `0.7851 -> 0.8213`, Brier `0.1868 -> 0.1678`)

---

## System Pipeline

```
MRNet Knee Exam (Axial + Coronal + Sagittal 2D slices, 224x224)
   │
   ├─► 2D Backbone (timm: ResNet50 / EfficientNet-B0 / Swin-Tiny / ViT-Base)
   │      └─► Masked Max-Pooling per Plane ─► Concatenate 3 Views ─► Dropout (0.3) ─► 3 Heads (Abnormal, ACL, Meniscus)
   │
   ├─► Ensemble Averaging (Top-3 Ensemble & Synopsis EffNet+ViT)
   ├─► Val-Only Platt Calibration & Youden Thresholds (zero Test leakage)
   ├─► Uncertainty Triage (Predictive Entropy / 4-Encoder Disagreement / MC Dropout)
   └─► Interactive Streamlit Demo (`app/app.py`) ──► [Next: Phase 5 Grad-CAM + Phase 6 LLM Summary]
```

---

## Repository Structure

```
orthovision/
├── configs/default.yaml          # FROZEN shared training protocol for all 4 encoders
├── splits/split_v1.json          # FROZEN split: 960 Train / 170 Val / 120 Test (MRNet valid)
├── src/orthovision/
│   ├── config.py                 # Config loader + path validation
│   ├── data/                     # MRNet dataset, preprocessing, slice collate, labels
│   ├── models/exam_model.py      # Shared 3-plane encoder + masked max-pool + dropout + 3 heads
│   ├── engine.py                 # AMP, grad accumulation (eff batch 8), BN freeze, weighted BCE
│   ├── eval/                     # AUC metrics, Temperature/Platt calibration, DeLong, ensembles
│   ├── uncertainty/              # MC Dropout (30 passes) & selective referral curves
│   ├── explain/                  # Phase 5 placeholder (Grad-CAM + region mapping)
│   └── llm/                      # Phase 6 placeholder (structured findings -> text)
├── scripts/
│   ├── check_env.py              # Environment check
│   ├── verify_data.py            # Verify MRNet dataset layout & labels
│   ├── eda.py                    # Exploratory data analysis
│   ├── make_splits.py            # Used once in Phase 1 (DO NOT re-run)
│   ├── train.py                  # Kaggle GPU training entrypoint (prefix with TQDM_DISABLE=1)
│   ├── evaluate_phase4.py        # Full Phase 4 evaluation (ensembles, calibration, DeLong, figures)
│   └── run_mc_dropout.py         # GPU MC Dropout (30 passes) script
├── app/
│   ├── demo_backend.py           # Modular Phase 4 backend + Phase 5/6 plug-in hooks
│   └── app.py                    # 4-tab Interactive Streamlit Dashboard
├── tests/                        # Unit & regression tests (`test_config`, `test_dataset`, `test_phase4`, ...)
├── docs/
│   ├── TEAM_GUIDE.md             # Plain-English project walkthrough & commit-by-commit explanation
│   ├── COMPUTE.md                # Kaggle/Colab compute guide
│   └── ROADMAP.md                # Phase breakdown
└── outputs/                      # Git-ignored (checkpoints, logs, prediction CSVs stored on Google Drive)
```

---

## Hard Project Rules (Never Break These)

1. **Never modify `configs/default.yaml`:** All 4 encoders must share identical preprocessing and training settings.
2. **Never regenerate `splits/split_v1.json`:** Seed `42` split (`960 / 170 / 120`) is frozen permanently. Do not run `scripts/make_splits.py`.
3. **Never tune on the Test set:** Checkpoints, Temperature/Platt calibration scalers, and Youden thresholds are fitted exclusively on the `170`-exam Validation split.
4. **Never commit raw MRNet data, `.pt` checkpoints, or `outputs/` to git:** Those are backed up on Google Drive (`OrthoVision Backups`).
5. **Kaggle training rule:** Always prefix training commands with `TQDM_DISABLE=1` and train on Kaggle Tesla T4 (see [`docs/COMPUTE.md`](docs/COMPUTE.md)).

---

## Quick Start & Demo Setup

Because Stanford MRNet medical scans are license-restricted (`data/README.md`), runtime demo files are split cleanly into two separate bundles (never committed to Git):
- **Public bundle (`phase4_outputs_public.zip`, `~53 KB` / `21` files):** only evaluation JSONs, training logs, and prediction CSVs (**zero MRI images**).
- **Private team-only bundle (`mrnet_test_previews_private.zip`, `72.83 MB` / `1,800` PNG slices):** 5 evenly spaced Axial / Coronal / Sagittal preview slices per held-out test exam (`1130`–`1249`), stored privately under team Google Drive (`OrthoVision Backups / Phase4_Demo_Pack / Private_Team_Only_MRNet_Previews`).

---

### Option A — Public Demo Setup (Examiners / External Users — No MRNet Dataset Required)

**Step 1 — Clone & install lightweight CPU dependencies (no PyTorch or GPU needed):**
```bash
git clone https://github.com/bl4zeistrying99/orthovision.git
cd orthovision
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install streamlit pandas numpy scipy scikit-learn matplotlib pyyaml pytest Pillow
```

**Step 2 — Add the public Phase 4 outputs bundle:**
Download `phase4_outputs_public.zip` (`~53 KB` zip / `0.17 MB` unpacked; contains only metrics, logs, and prediction CSVs — no MRI scans) and unzip it directly into the `orthovision/` repo root (creating `outputs/logs/` and `outputs/predictions/`):
```powershell
# Windows (PowerShell, inside orthovision/)
Expand-Archive -Path "$HOME\Downloads\phase4_outputs_public.zip" -DestinationPath . -Force
```
```bash
# Linux / macOS (inside orthovision/)
unzip -o ~/Downloads/phase4_outputs_public.zip -d .
```

**Step 3 — Verify tests & launch the interactive dashboard:**
```bash
python tests/test_phase4.py      # 6 CPU tests pass (1 GPU/torch test auto-skips on CPU)
streamlit run app/app.py         # opens all 4 interactive Phase 4 dashboard tabs
```

**Step 4 *(Optional, for authorized Stanford MRNet holders)* — Enable knee MRI photos:**  
Place the official `valid/` directory under `data/raw/MRNet-v1.0/valid/` (or set `ORTHOVISION_DATA_ROOT` / `configs/local.yaml` -> `paths.data_root`). All 3 MRI views (`Axial`, `Coronal`, `Sagittal`) will appear automatically in **Exam Explorer**.

---

### Option B — Team Setup (`MP2026CSE223` — Full Demo with 3-Plane Knee MRI Photos, No `5.7 GB` Download)

1. Complete **Option A (Steps 1–2)** above so `outputs/logs/` and `outputs/predictions/` are in place.
2. Download `mrnet_test_previews_private.zip` (`72.83 MB`, `1,800` PNG slices for all `120` test exams) from our team's private Google Drive (`OrthoVision Backups / Phase4_Demo_Pack / Private_Team_Only_MRNet_Previews`) and unzip directly into `orthovision/` (creates git-ignored `data/processed/exam_previews/`):
   ```powershell
   # Windows (PowerShell, inside orthovision/)
   Expand-Archive -Path "$HOME\Downloads\mrnet_test_previews_private.zip" -DestinationPath . -Force
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app/app.py
   ```
   All `120` held-out test exams (`1130`–`1249`) now display interactive **Axial**, **Coronal**, and **Sagittal** knee MRI slices (`5` slices per view) automatically.

---

### Expected Folder Structure After Unzipping (External User vs. Team Member)

Check that your folders inside `orthovision/` match one of these two layouts:

#### 1. External User / Examiner (Public Bundle Only — `phase4_outputs_public.zip`, `52.8 KB`)
Runs all 4 dashboard tabs cleanly with **zero MRI scan files** (the MRI viewer displays clean placeholder cards unless you add your own authorized MRNet `valid/` scans below):

```text
orthovision/
├── outputs/
│   ├── logs/
│   │   ├── phase4_metrics.json
│   │   ├── resnet50_seed42/         (log.csv, final_metrics.json)
│   │   ├── efficientnet_b0_seed42/  (log.csv, final_metrics.json)
│   │   ├── swin_seed42/             (log.csv, final_metrics.json)
│   │   └── vit_seed42/              (log.csv, final_metrics.json)
│   └── predictions/
│       └── {resnet50,efficientnet_b0,swin,vit,effnet_vit,top3_ensemble}_seed42_{val,test}.csv
└── data/
    ├── processed/                   (empty — placeholder MRI cards shown)
    └── raw/                         (empty; or put MRNet-v1.0/valid/{axial,coronal,sagittal}/1130.npy..1249.npy if authorized)
```

#### 2. Team Member (`MP2026CSE223` — Public + Private Team Bundle `mrnet_test_previews_private.zip`, `72.8 MB`)
Includes everything in External User above **plus** the lightweight PNG knee MRI slices (`1,800` PNGs total) so **Exam Explorer** shows real MRI photos with slice sliders:

```text
orthovision/
├── outputs/                         (same 21 files from phase4_outputs_public.zip)
└── data/
    └── processed/
        └── exam_previews/           (git-ignored — created by unzipping mrnet_test_previews_private.zip)
            ├── axial/               (1130_00.png ... 1249_04.png  — 600 PNGs)
            ├── coronal/             (1130_00.png ... 1249_04.png  — 600 PNGs)
            └── sagittal/            (1130_00.png ... 1249_04.png  — 600 PNGs)
```

> **Common Mistake Check:** Always unzip directly into the **`orthovision/`** root folder (`-DestinationPath .` / `-d .`). Never double-nest folders like `outputs/outputs/...` or `data/processed/exam_previews/data/...`, or Streamlit will report missing files.

---

### Option C — Full Training / Inference Environment (Kaggle Tesla T4 or Local CUDA GPU)

```bash
# 1) Install PyTorch first from https://pytorch.org/get-started/locally/
# 2) Then install full dependencies:
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_env.py
pytest -q
# Re-run full Phase 4 evaluation anytime from outputs/:
python scripts/evaluate_phase4.py
```

Detailed team onboarding, plain-English explanations of all Phase 0–4 techniques, and what every GitHub commit changed are in [`docs/TEAM_GUIDE.md`](docs/TEAM_GUIDE.md).
