"""Prediction CSV utilities and probability-averaging ensembles."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions_csv(
    path: str | Path,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load `*_val.csv` or `*_test.csv` sorted by exam `id` -> `(ids, y_true, y_prob)`."""
    df = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    ids = df["id"].to_numpy(dtype=np.int64)
    y_true = df[[f"y_{lab}" for lab in labels]].to_numpy(dtype=np.int64)
    y_prob = df[[f"p_{lab}" for lab in labels]].to_numpy(dtype=np.float64)
    return ids, y_true, y_prob


def save_predictions_csv(
    path: str | Path,
    ids: np.ndarray,
    probs: np.ndarray,
    ys: np.ndarray,
    labels: list[str],
) -> Path:
    """Save predictions in the exact CSV format produced by `orthovision.engine`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"y_{lab}" for lab in labels] + [f"p_{lab}" for lab in labels])
        for eid, y_row, p_row in zip(ids, ys, probs):
            w.writerow([int(eid)] + [int(v) for v in y_row] + [f"{float(v):.6f}" for v in p_row])
    return out


def average_ensemble(
    csv_paths: list[str | Path],
    labels: list[str],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted probability average across verified exam-aligned prediction CSVs."""
    if not csv_paths:
        raise ValueError("csv_paths must contain at least one prediction file")

    loaded = [load_predictions_csv(p, labels) for p in csv_paths]
    ref_ids, ref_y, _ = loaded[0]
    for path, (ids, ys, _) in zip(csv_paths[1:], loaded[1:]):
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"Exam ID mismatch in ensemble member: {path}")
        if not np.array_equal(ys, ref_y):
            raise ValueError(f"Ground-truth label mismatch in ensemble member: {path}")

    if weights is None:
        w = np.full(len(loaded), 1.0 / len(loaded), dtype=np.float64)
    else:
        if len(weights) != len(loaded):
            raise ValueError("weights length must match csv_paths length")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        w = w / w.sum()

    prob_stack = np.stack([item[2] for item in loaded], axis=0)  # (M, N, L)
    p_ens = np.tensordot(w, prob_stack, axes=(0, 0))
    return ref_ids, ref_y, p_ens
