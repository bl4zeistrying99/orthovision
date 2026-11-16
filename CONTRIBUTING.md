# Team workflow (5 minutes, saves hours)

1. `main` is always runnable. Nobody pushes directly to it.
2. One branch per task: `git checkout -b <yourname>/<short-task>`, e.g. `amey/gradcam-regions`.
3. Open a Pull Request, get ONE teammate to look, then merge. Keep PRs small (< ~300 lines).
4. `git pull origin main` before you start work each day.
5. Before pushing: `pytest -q` and `ruff check src tests scripts`.
6. Never commit: data, `.npy`, checkpoints (`.pt`), `.env`, `configs/local.yaml` (already in .gitignore).
7. Hyperparameters live in `configs/default.yaml`, never hard-coded. Do not change the shared
   protocol without telling everyone: it would invalidate the model comparison.
8. Every experiment uses `set_seed(cfg["seed"])`. Log results to `outputs/logs/` as CSV/JSON.
9. Checkpoints go to shared storage (Google Drive folder / Kaggle dataset), named
   `<model>_seed<seed>_<date>.pt`.
10. Interfaces first: agree function signatures (e.g. `predict(exam) -> dict`) before two
    people build against each other.
