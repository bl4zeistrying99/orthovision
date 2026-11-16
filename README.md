# OrthoVision AI
Explainable and uncertainty-aware knee MRI analysis (MRNet) | B.Tech CSE, Graphic Era | Team MP2026CSE223

## Quick start (every teammate, once)
```bash
git clone <repo-url> && cd orthovision
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# 1) PyTorch first, command from https://pytorch.org/get-started/locally/ (match your CUDA)
# 2) then:
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_env.py       # must end with "Environment OK"
pytest -q                         # must pass
```
No local GPU? Use Kaggle/Colab: see `docs/COMPUTE.md`.

## Data
See `data/README.md`. Data is never committed to git. Then run `python scripts/verify_data.py`.

## Layout
```
configs/default.yaml   ONE shared protocol for all models (identical-protocol comparison)
src/orthovision/
  data/         preprocess, labels/splits, MRNetDataset + collate
  models/       encoders + slice->series->exam aggregation      (Phase 2-3)
  eval/         AUC, calibration (ECE, Brier), DeLong            (Phase 3-4)
  uncertainty/  MC Dropout                                       (Phase 4)
  explain/      Grad-CAM + region mapping                        (Phase 5)
  llm/          structured findings -> text                      (Phase 6)
app/            Streamlit dashboard                              (Phase 6)
scripts/        check_env, verify_data, (train, evaluate later)
tests/          pytest
splits/         frozen split_v1.json (committed, never regenerate)
```
Team workflow: `CONTRIBUTING.md`. Phases and owners: `docs/ROADMAP.md`.
