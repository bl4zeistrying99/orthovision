# Roadmap and suggested ownership (edit names/owners as a team)

| Phase | Deliverable | Suggested owner |
|---|---|---|
| 0 | Environment, repo, config, data layer, tests (DONE) | all: run setup |
| 1 | Data verified, EDA notebook (class balance, slice counts), splits frozen | Member A |
| 2 | Generic exam model (timm backbone + max-pool + 3 heads), train loop, ResNet50 baseline | Member A + B |
| 3 | EfficientNet-B0, ViT, Swin training; ensemble; AUC/PR/F1 eval | one encoder each |
| 4 | Calibration (ECE/Brier/reliability), MC Dropout, DeLong | Member C |
| 5 | Grad-CAM, heatmap to region mapping | Member C |
| 6 | LLM summary layer, Streamlit dashboard | Member D |
| 7 | Ablations, stretch goals, report | all |

Critical path: Phase 2 unblocks everything. Phases 4-6 can start in parallel on a stub model
as soon as the `predict()` interface is agreed.
