"""MC Dropout epistemic uncertainty and selective radiologist-referral utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def enable_head_dropout(model: Any) -> None:
    """Freeze backbone & BatchNorm in eval mode while enabling ONLY `model.dropout`."""
    model.eval()
    if not hasattr(model, "dropout"):
        raise AttributeError("Expected ExamModel with a `.dropout` layer")
    model.dropout.train(True)


def encode_pooled_features(model: Any, series: dict, mask: dict):
    """Run shared 2D slice encoder + masked_pool once per exam -> (B, n_planes * F)."""
    import torch

    from ..models.exam_model import masked_pool

    pooled = []
    for p in model.planes:
        x = series[p]
        b, s = x.shape[:2]
        f = model._encode(x.flatten(0, 1)).reshape(b, s, -1)
        pooled.append(masked_pool(f, mask[p], model.pooling))
    return torch.cat(pooled, dim=1)


def _binary_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))


def summarize_mc_samples(
    mc_probs: np.ndarray,
    ids: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    eps: float = 1e-7,
) -> dict[str, np.ndarray]:
    """Summarize (N, T, L) MC probability passes into mean, std, entropy, and BALD MI."""
    samples = np.asarray(mc_probs, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError(f"Expected mc_probs of shape (N, n_passes, n_labels), got {samples.shape}")

    p_mean = samples.mean(axis=1)
    p_std = samples.std(axis=1, ddof=0)
    pred_entropy = _binary_entropy(p_mean, eps=eps)
    expected_entropy = _binary_entropy(samples, eps=eps).mean(axis=1)
    mutual_info = np.maximum(0.0, pred_entropy - expected_entropy)

    out: dict[str, np.ndarray] = {
        "mc_probs": samples,
        "p_mean": p_mean,
        "p_std": p_std,
        "predictive_entropy": pred_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_info,
    }
    if ids is not None:
        out["id"] = np.asarray(ids, dtype=np.int64)
    if y_true is not None:
        out["y_true"] = np.asarray(y_true, dtype=np.int64)
    return out


def mc_predict(
    model: Any,
    loader: Any,
    device: Any,
    n_passes: int = 30,
    seed: int = 42,
    amp: bool = True,
) -> dict[str, np.ndarray]:
    """Run T stochastic head-only MC Dropout passes using frozen backbone features."""
    import torch
    from tqdm import tqdm

    if n_passes < 2:
        raise ValueError("n_passes must be >= 2 for uncertainty estimation")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    enable_head_dropout(model)
    ids: list[int] = []
    ys: list[np.ndarray] = []
    mc_batches: list[np.ndarray] = []

    use_amp = bool(amp) and getattr(device, "type", str(device)) == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"mc_predict (T={n_passes})", leave=False):
            series = {p: x.to(device, non_blocking=True) for p, x in batch["series"].items()}
            mask = {p: m.to(device, non_blocking=True) for p, m in batch["mask"].items()}

            with torch.autocast(device_type="cuda", enabled=use_amp):
                feats = encode_pooled_features(model, series, mask)

            feats_f = feats.float()
            pass_probs = []
            for _ in range(n_passes):
                logits = model.head(model.dropout(feats_f))
                pass_probs.append(torch.sigmoid(logits).cpu().numpy())

            # stack passes -> (B, T, L)
            mc_batches.append(np.stack(pass_probs, axis=1))
            ys.append(batch["y"].numpy())
            ids.extend(int(i) for i in batch["id"])

    mc_probs = np.concatenate(mc_batches, axis=0)
    y_true = np.concatenate(ys, axis=0)
    return summarize_mc_samples(mc_probs, ids=np.asarray(ids), y_true=y_true)


def simulate_head_mc_from_probs(
    y_prob: np.ndarray,
    dropout_p: float = 0.3,
    n_passes: int = 30,
    feature_dim: int = 6144,
    seed: int = 42,
) -> np.ndarray:
    """Analytical local proxy for inverted head-dropout variance when `.pt` weights are absent.

    NOTE: This is a local surrogate approximation derived from output logits rather than true
    checkpoint forward passes; use `scripts/run_mc_dropout.py` on GPU for exact weights-based MC.
    """
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    z = np.log(p_arr / (1.0 - p_arr))
    if dropout_p <= 0.0:
        return np.repeat(p_arr[:, None, :], n_passes, axis=1)
    if dropout_p >= 1.0:
        raise ValueError("dropout_p must be in [0.0, 1.0)")

    d_eff = max(1.0, float(feature_dim) / 128.0)
    sigma_z = np.sqrt((dropout_p / (1.0 - dropout_p)) * (z ** 2 + 1.0) / d_eff)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((z.shape[0], n_passes, z.shape[1]))
    z_mc = z[:, None, :] + sigma_z[:, None, :] * noise
    return expit(z_mc)


def selective_referral_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    fractions: tuple[float, ...] | list[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Measure accuracy, Brier, and ROC-AUC on retained exams after referring top-k% uncertain."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if not (len(y) == len(p) == len(u)):
        raise ValueError("y_true, y_prob, and uncertainty must have identical length")

    n_total = len(y)
    order = np.argsort(u, kind="mergesort")  # ascending uncertainty: lowest kept first
    rows: list[dict[str, float]] = []

    for frac in fractions:
        n_refer = int(round(frac * n_total))
        n_keep = max(1, n_total - n_refer)
        keep_idx = order[:n_keep]
        yk, pk, uk = y[keep_idx], p[keep_idx], u[keep_idx]

        acc = float(((pk >= threshold).astype(np.int64) == yk).mean())
        brier = float(np.mean((pk - yk) ** 2))
        auc = float(roc_auc_score(yk, pk)) if len(np.unique(yk)) > 1 else float("nan")
        rows.append(
            {
                "referral_fraction": float(frac),
                "kept_count": int(n_keep),
                "referred_count": int(n_total - n_keep),
                "coverage": float(n_keep / n_total),
                "accuracy": acc,
                "roc_auc": auc,
                "brier": brier,
                "mean_uncertainty_kept": float(uk.mean()),
            }
        )
    return rows
