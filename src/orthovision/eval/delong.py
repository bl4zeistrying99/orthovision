"""Statistical evaluation: paired DeLong ROC-AUC test, stratified bootstrap CIs, and thresholds."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, roc_curve


def _delong_structural(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute exact Mann-Whitney / DeLong AUC and structural components (V10, V01)."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = int(pos_mask.sum())
    n = int(neg_mask.sum())
    if m < 2 or n < 2:
        raise ValueError(f"DeLong test requires at least 2 positives and 2 negatives (got m={m}, n={n})")

    x = scores[pos_mask]
    y = scores[neg_mask]
    tx = rankdata(x, method="average")
    ty = rankdata(y, method="average")
    txy = rankdata(np.concatenate([x, y]), method="average")

    v10 = (txy[:m] - tx) / n
    v01 = 1.0 - (txy[m:] - ty) / m
    return float(v10.mean()), v10, v01


def delong_roc_test(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
) -> dict[str, float]:
    """Two-sided paired DeLong test comparing ROC-AUC(p_a) vs ROC-AUC(p_b) on identical labels."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    sa = np.asarray(p_a, dtype=np.float64).ravel()
    sb = np.asarray(p_b, dtype=np.float64).ravel()
    if not (len(y) == len(sa) == len(sb)):
        raise ValueError("y_true, p_a, and p_b must have identical length")

    m = int((y == 1).sum())
    n = int((y == 0).sum())
    auc_a, v10_a, v01_a = _delong_structural(y, sa)
    auc_b, v10_b, v01_b = _delong_structural(y, sb)

    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    sigma = s10 / m + s01 / n
    var = float(sigma[0, 0] + sigma[1, 1] - 2.0 * sigma[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    diff = float(auc_a - auc_b)

    if se <= 1e-15:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = float(diff / se)
        p_val = float(2.0 * norm.sf(abs(z_stat)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": diff,
        "se": se,
        "z": z_stat,
        "p_value": p_val,
    }


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Stratified percentile bootstrap confidence interval for ROC-AUC."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Stratified bootstrap requires both positive and negative examples")

    rng = np.random.default_rng(seed)
    m, n = len(pos_idx), len(neg_idx)
    y_boot = np.concatenate([np.ones(m, dtype=np.int64), np.zeros(n, dtype=np.int64)])
    aucs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        p_pos = p[rng.choice(pos_idx, size=m, replace=True)]
        p_neg = p[rng.choice(neg_idx, size=n, replace=True)]
        aucs[b] = roc_auc_score(y_boot, np.concatenate([p_pos, p_neg]))

    alpha = (1.0 - ci) * 50.0
    return float(np.percentile(aucs, alpha)), float(np.percentile(aucs, 100.0 - alpha))


def youden_optimal_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Select decision threshold maximizing Youden's J (sensitivity + specificity - 1) on VAL only."""
    y = np.asarray(y_val, dtype=np.int64).ravel()
    p = np.asarray(p_val, dtype=np.float64).ravel()
    fpr, tpr, thresholds = roc_curve(y, p)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    thr = float(thresholds[best_idx])
    if not np.isfinite(thr):
        thr = 1.0
    return float(np.clip(thr, 0.0, 1.0))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary clinical classification metrics at a fixed probability threshold."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    pred = (p >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2.0 * tp, 2 * tp + fp + fn)
    acc = _safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
