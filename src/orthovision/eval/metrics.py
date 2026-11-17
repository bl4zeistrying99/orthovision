from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Per-label and macro ROC-AUC. y_*: (N, n_labels). NaN if a label has one class only."""
    out = {}
    for j, lab in enumerate(labels):
        col = y_true[:, j]
        out[f"auc_{lab}"] = (
            float(roc_auc_score(col, y_prob[:, j])) if len(np.unique(col)) > 1 else float("nan")
        )
    vals = [out[f"auc_{lab}"] for lab in labels]
    out["auc_macro"] = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    return out
