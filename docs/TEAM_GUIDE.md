# OrthoVision AI — Team Guide & GitHub Commit History

> **Research Prototype Only — Not for Clinical Diagnosis.**  
> **Team ID:** `MP2026CSE223` · **Guide:** Mr. Vishal Trivedi · Graphic Era Deemed to be University, Dehradun  
> **Team Members:** Bhawesh Gunjiyal, Rudra Pratap Singh, Amey Jhaldiyal, Daksh Choudhary

This guide helps every teammate understand **what our project does**, **what each GitHub commit added and why**, **our verified Phase 0–4 results**, and **how to prepare for the Phase-I Examination (Dec 7–11, 2026)**.

---

## 1. The Project in 2 Minutes

**What problem do we solve?**  
A knee MRI scan is not one picture — it is three separate stacks of grayscale slices (**Axial**, **Coronal**, and **Sagittal** views, ~17–61 slices each). A radiologist must check all three views for abnormalities, ACL (Anterior Cruciate Ligament) tears, and meniscus tears.

**What does OrthoVision AI do?**  
1. **Reads all 3 MRI views together** for one patient exam (`224 x 224` resized slices).
2. **Extracts slice features** using a shared 2D pretrained vision backbone (`ResNet50`, `EfficientNet-B0`, `Swin-Tiny`, or `ViT-Base`) via `timm`.
3. **Combines all slices per view** using **masked max-pooling** (finding the strongest pathology signal across variable-length slice stacks without padding distorting the result), joins all 3 views, applies **Dropout (`0.3`)**, and predicts 3 probabilities: **Abnormal**, **ACL Tear**, and **Meniscus Tear**.
4. **Calibrates confidence & quantifies uncertainty** (Phase 4): converts raw scores into trustworthy probabilities (`Platt Scaling` fitted only on Validation) and flags the most uncertain exams for priority radiologist review (**Selective Triage**).
5. **Next phases (5 & 6):** Visual **Grad-CAM** heatmaps highlighting *where* in the slice the model looked + **LLM summary** turning structured outputs into plain text.

**Dataset & Frozen Split (`splits/split_v1.json`, Seed `42`):**
- Source: **Stanford MRNet v1.0** (`1,250` public exams total — Stanford's true test set is private, so official MRNet `valid` is our locked **Test** set).
- **Train:** `960` exams · **Validation:** `170` exams (patient/exam-level stratified) · **Test:** `120` exams.
- **Golden Rule:** We **never** tune thresholds, scalers, or checkpoints on the 120 Test exams.

---

## 2. GitHub Commit History Explained (Every Commit So Far)

Here is what every commit on `main` changed, what libraries/methods were used, and **why** we chose them:

| # | Commit | Date | Files | Summary |
|---|---|---|---|---|
| 1 | `8012ca5` | Nov 16, 2026 | 41 files | **Phase 0: project scaffold** — clean folder layout, config, data loader, seed helper, and unit tests |
| 2 | `b6f3e7e` | Nov 17, 2026 | 16 files | **Addition of Splits.V1** — locked patient-level `960 / 170 / 120` split (`splits/split_v1.json`) |
| 3 | `7f05efe` | Nov 17, 2026 | 11 files | **Phase 2** — `ExamModel`, memory-safe Kaggle GPU trainer, baseline training pipeline & tests |
| 4 | `1c30ee5` | Nov 17, 2026 | 15 files | **chore: untrack pycache/egg-info & ignore predictions** — repo hygiene before Phase 3 Kaggle runs |
| — | *(Kaggle)* | Nov 17–19 | *(Drive)* | **Phase 3 runs (no new code commit needed):** trained all 4 encoders using `scripts/train.py` + `configs/default.yaml`; saved weights & prediction CSVs to Google Drive |
| 5 | `9a893b1` | Nov 20, 2026 | 9 files (+1,525 lines) | **Phase 4** — ensembles, Temperature & Platt calibration, DeLong tests, MC Dropout / triage module & tests |

### Commit 1 — `8012ca5` · `Phase 0: project scaffold`
- **What changed:** Built the repository structure (`configs/default.yaml`, `src/orthovision/config.py`, `data/{dataset,labels,preprocess}.py`, `utils/seed.py`, `scripts/{check_env,verify_data,eda,make_splits}.py`, unit tests, `docs/{COMPUTE,ROADMAP}.md`).
- **What is used:** `PyYAML`, `NumPy`, `Pandas`, `scikit-learn`, `PyTorch Dataset` abstractions, `pytest`.
- **Why:** Putting preprocessing (`224x224` resize, ImageNet normalization), config (`seed: 42`), and tests in place from Day 1 ensures all 4 encoders are compared **fairly under identical conditions**, with zero hidden differences across teammates' laptops or Kaggle notebooks.

### Commit 2 — `b6f3e7e` · `Addition of Splits.V1`
- **What changed:** Committed `splits/split_v1.json` (plus `.pyc`/`.egg-info` artifacts that were later cleaned in Commit 4).
- **What is used:** Stratified exam-level train/val split generated once with Seed `42`.
- **Why:** In medical imaging, splitting by *slice* instead of by *exam* leaks the same patient into both training and validation, causing fake high scores. Locking `split_v1.json` guarantees every encoder sees the exact same `960 / 170 / 120` exams.

### Commit 3 — `7f05efe` · `Phase 2`
- **What changed:** Implemented the core deep-learning architecture and training engine (`src/orthovision/models/exam_model.py`, `src/orthovision/engine.py`, `src/orthovision/eval/metrics.py`, `scripts/train.py`, `configs/default.yaml` tuning params, `tests/test_{metrics,model}.py`).
- **What is used & why:**
  - `timm` shared backbone across `(axial, coronal, sagittal)` + **masked max-pooling** -> handles variable slice counts per knee scan cleanly.
  - **Weighted BCE (`use_pos_weight: true`)** -> corrects severe MRNet class imbalance (few positive ACL tears vs many abnormal scans).
  - **AMP (`fp16`), gradient checkpointing, `max_slices_train: 24`, `accum_steps: 8`, `freeze_bn: true`** -> lets heavy models like `ViT-Base` (`86M` params) and `Swin-Tiny` fit inside Kaggle's free **15 GB Tesla T4 GPU** while simulating an effective batch size of `8` exams and stabilizing BatchNorm on small batches.

### Commit 4 — `1c30ee5` · `chore: ignore outputs/predictions and untrack pycache/egg-info`
- **What changed:** Updated `.gitignore` to exclude `outputs/predictions/*` and removed accidentally tracked `__pycache__/*.pyc` and `orthovision.egg-info/` files from Git tracking.
- **Why:** Large prediction CSVs, `.pt` checkpoints, and compiled bytecode bloat GitHub and cause merge conflicts. Weights and prediction outputs live safely in Google Drive (`OrthoVision Backups`), keeping GitHub fast and clean.
- **Note on Phase 3 (Nov 17–19):** Because Phase 2 code (`scripts/train.py` + `configs/default.yaml`) was completely generic (`--model resnet50 / efficientnet_b0 / swin / vit`), **all 4 encoders were trained on Kaggle T4 without changing a single line of model code**.

### Commit 5 — `9a893b1` · `Phase 4`
- **What changed:** Added our full evaluation, calibration, statistical testing, and uncertainty triage suite:
  - `src/orthovision/eval/ensemble.py` — combines model probabilities (`Top-3 Ensemble` and `EffNet+ViT`).
  - `src/orthovision/eval/calibration.py` — **ECE, MCE, Brier score, NLL, Temperature Scaling, Platt Scaling**, and reliability diagram plotter.
  - `src/orthovision/eval/delong.py` — **Paired DeLong ROC-AUC significance test**, 2,000-iteration stratified bootstrap 95% CIs, Youden's J thresholding (Val-only), and Sensitivity/Specificity/F1 metrics.
  - `src/orthovision/uncertainty/mc_dropout.py` — **MC Dropout (30 passes)**, predictive entropy, BALD mutual information, and **selective radiologist referral curves**.
  - `scripts/evaluate_phase4.py`, `scripts/run_mc_dropout.py`, `tests/test_phase4.py` — automated pipeline producing `outputs/logs/phase4_metrics.json`, publication figures, and 6 CPU-verified unit tests.
- **Why:** In healthcare AI, raw ROC-AUC is not enough. Weighted BCE stretches probability scores away from true clinical risk (`calibration` fixes this), a small test set (`n = 120`) needs a paired statistical significance test (`DeLong`) before claiming one model beats another, and safe deployment needs the model to say **"I am unsure — send this scan to a radiologist"** (`selective referral`).

---

## 3. Measured Phase 3 & Phase 4 Scoreboard (Test Split, `n = 120`)

| Model / Ensemble | Val Macro AUC | Test Macro AUC | Abnormal AUC | ACL AUC | Meniscus AUC | Raw ECE | Platt ECE | Youden Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Top-3 Ensemble** (`R50+Eff+Swin`) | 0.8815 | **0.9000** | 0.9516 | **0.9632** | **0.7851** | **0.0563** | 0.0772 | **0.8408** |
| **ResNet50** (Best Ep 16) | **0.8893** | 0.8944 | **0.9571** | 0.9610 | 0.7653 | 0.1056 | 0.0933 | 0.8283 |
| **Swin-Tiny** (Best Ep 18) | 0.8549 | 0.8756 | 0.9406 | 0.9198 | 0.7664 | 0.1203 | 0.1054 | 0.8078 |
| **EfficientNet-B0** (Best Ep 9) | 0.8509 | 0.8716 | 0.9183 | 0.9287 | 0.7677 | 0.1059 | 0.1023 | 0.8154 |
| **EffNet-B0 + ViT** | 0.8349 | 0.8684 | 0.9112 | 0.9122 | 0.7820 | 0.0899 | **0.0864** | 0.8184 |
| **ViT-Base** (Best Ep 14) | 0.8037 | 0.8158 | 0.8684 | 0.8363 | 0.7428 | 0.0888 | 0.1061 | 0.7281 |

### 4 Talking Points Every Teammate Must Know for Phase-I Viva
1. **Why did ResNet50 (`0.8944`) beat pure ViT-Base (`0.8158`) on MRNet?**  
   With only `960` training exams, CNN inductive bias (translation equivariance + local edges) learns stronger representations than a data-hungry pure Transformer (`ViT-Base`). Combining Swin-Tiny + CNNs in the **Top-3 Ensemble** achieves our best **Macro AUC (`0.9000`)**.
2. **Why is Meniscus AUC lower (`0.7851`) than Abnormal (`0.9516`) and ACL (`0.9632`)?**  
   Meniscus tears are small, subtle, and focal across 2D slices (`224x224` with max-pooling), whereas ACL tears and general abnormalities affect broader anatomical structures. This directly motivates **Phase 5 (Grad-CAM + region localization)**.
3. **Why did we run Platt Scaling + DeLong's Test?**  
   Weighted BCE distorts output probabilities; Validation-fitted calibration aligns predicted risk with observed frequency while mathematically preserving ROC-AUC (`diff <= 1e-6`). DeLong proves `Top-3 Ensemble`'s ACL gain over `EfficientNet-B0` (`p = 0.0315`), `Swin-Tiny` (`p = 0.0357`), and `ViT-Base` (`p = 0.0002`) is statistically real, not luck.
4. **How does uncertainty triage improve safety?**  
   Referring the top 25% most uncertain exams (`30/120`, ranked by Top-3 predictive entropy) lifts accuracy on retained exams: Abnormal **`85.0% -> 95.6%`**, ACL **`88.3% -> 96.7%`** (`AUC 0.9905`), Meniscus **`71.7% -> 75.6%`**.

---

## 4. What Happens Next & Milestones

| Date (Tentative Dept. Calendar) | Milestone | Target Goal |
|---|---|---|
| **Nov 20, 2026 (Now)** | Interactive Phase 4 Streamlit Demo (`streamlit run app/app.py`) | Mentor review (`Mr. Vishal Trivedi`) |
| **Nov 30, 2026** | Research Paper Part 1 + Part 2 (Intro, Lit Survey, Methodology) | Draft complete from Phase 0–4 pipeline |
| **Dec 7–11, 2026** | **Major Project Phase-I Examination** | Live demo + Phase 3/4 benchmark + Viva |
| **Dec 2026 – Jan 2027** | **Phase 5 (Grad-CAM) & Phase 6 (LLM Summary + Final Report)** | Complete early (targeting **before Feb 2027**) |
| **Feb 7, 2027** | Full paper submitted to Scopus-indexed conference/journal | Ready ahead of deadline |
