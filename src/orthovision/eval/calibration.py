"""Probability calibration metrics, scalers (Validation-fitted only), and reliability plots."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


def prob_to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Invert sigmoid safely into log-odds logits."""
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return expit(np.asarray(z, dtype=np.float64))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error (lower is better; 0.0 is perfect)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def nll_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Binary negative log-likelihood / log loss (lower is better)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute equal-width Expected/Maximum Calibration Error and bin statistics."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    n_total = len(y)
    ece = 0.0
    mce = 0.0
    bin_counts: list[int] = []
    bin_acc: list[float | None] = []
    bin_conf: list[float | None] = []

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (p >= lo) & (p <= hi) if b == 0 else (p > lo) & (p <= hi)
        count = int(in_bin.sum())
        bin_counts.append(count)
        if count == 0:
            bin_acc.append(None)
            bin_conf.append(None)
            continue
        acc = float(y[in_bin].mean())
        conf = float(p[in_bin].mean())
        gap = abs(acc - conf)
        ece += (count / n_total) * gap
        mce = max(mce, gap)
        bin_acc.append(acc)
        bin_conf.append(conf)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_edges": edges.tolist(),
        "bin_counts": bin_counts,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
    }


class _BaseLogitScaler:
    """Shared validation/transform scaffolding for monotonic per-label logit scalers."""

    method: str = "base"

    def __init__(self, labels: list[str] | None = None):
        self.labels_: list[str] = list(labels) if labels is not None else []
        self._fitted = False

    def _validate_2d(self, p: np.ndarray) -> np.ndarray:
        arr = np.asarray(p, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array (N, n_labels), got shape {arr.shape}")
        if self._fitted and arr.shape[1] != len(self.labels_):
            raise ValueError(
                f"Expected {len(self.labels_)} label columns {self.labels_}, got {arr.shape[1]}"
            )
        return arr

    def fit(self, y_val: np.ndarray, p_val: np.ndarray, labels: list[str] | None = None):
        """Fit calibration parameters strictly on the validation split."""
        y_arr = self._validate_2d(y_val)
        p_arr = self._validate_2d(p_val)
        if y_arr.shape != p_arr.shape:
            raise ValueError(f"Shape mismatch: y_val {y_arr.shape} vs p_val {p_arr.shape}")
        if labels is not None:
            self.labels_ = list(labels)
        elif not self.labels_:
            self.labels_ = [f"label_{j}" for j in range(y_arr.shape[1])]
        if len(self.labels_) != y_arr.shape[1]:
            raise ValueError("Length of labels must match number of columns")

        logits = prob_to_logit(p_arr)
        for j, lab in enumerate(self.labels_):
            self._fit_column(lab, y_arr[:, j], logits[:, j])
        self._fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Apply validation-fitted calibration to probabilities (preserves AUC when slope > 0)."""
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before calling transform()")
        p_arr = self._validate_2d(p)
        logits = prob_to_logit(p_arr)
        out = np.empty_like(p_arr, dtype=np.float64)
        for j, lab in enumerate(self.labels_):
            out[:, j] = self._transform_column(lab, logits[:, j])
        return out

    def _fit_column(self, label: str, y_col: np.ndarray, z_col: np.ndarray) -> None:
        raise NotImplementedError

    def _transform_column(self, label: str, z_col: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TemperatureScaler(_BaseLogitScaler):
    """Per-label 1-parameter temperature scaling: p_cal = sigmoid(logit(p) / T), T > 0."""

    method = "temperature"

    def __init__(self, labels: list[str] | None = None, bounds: tuple[float, float] = (0.05, 10.0)):
        super().__init__(labels)
        self.bounds = bounds
        self.temperatures_: dict[str, float] = {}

    def _fit_column(self, label: str, y_col: np.ndarray, z_col: np.ndarray) -> None:
        def loss(temp: float) -> float:
            return nll_score(y_col, sigmoid(z_col / temp))

        res = minimize_scalar(loss, bounds=self.bounds, method="bounded")
        self.temperatures_[label] = float(res.x)

    def _transform_column(self, label: str, z_col: np.ndarray) -> np.ndarray:
        return sigmoid(z_col / self.temperatures_[label])

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperatures": dict(self.temperatures_)}


class PlattScaler(_BaseLogitScaler):
    """Per-label 2-parameter Platt / logistic calibration: p_cal = sigmoid(a * logit(p) + b), a > 0.

    Training MRNet with `use_pos_weight: true` (e.g. abnormal ~0.23, acl ~4.93, meniscus ~1.82)
    shifts raw logits away from true prevalence log-odds (b != 0) in addition to mis-scaling
    confidence (a != 1). Platt scaling corrects both slope `a` and intercept `b` on the
    validation split while strictly preserving rank order / ROC-AUC (because a > 0).
    """

    method = "platt"

    def __init__(self, labels: list[str] | None = None, reg: float = 1e-4):
        super().__init__(labels)
        self.reg = reg
        self.params_: dict[str, dict[str, float]] = {}

    def _fit_column(self, label: str, y_col: np.ndarray, z_col: np.ndarray) -> None:
        def loss(theta: np.ndarray) -> float:
            a, b = float(theta[0]), float(theta[1])
            nll = nll_score(y_col, sigmoid(a * z_col + b))
            penalty = self.reg * ((a - 1.0) ** 2 + b ** 2)
            return nll + penalty

        res = minimize(
            loss,
            x0=np.array([1.0, 0.0], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0.05, 10.0), (-10.0, 10.0)],
        )
        self.params_[label] = {"a": float(res.x[0]), "b": float(res.x[1])}

    def _transform_column(self, label: str, z_col: np.ndarray) -> np.ndarray:
        pr = self.params_[label]
        return sigmoid(pr["a"] * z_col + pr["b"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": {k: dict(v) for k, v in self.params_.items()},
        }


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute per-label and macro ECE, MCE, Brier score, and NLL."""
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    per_label: dict[str, Any] = {}
    for j, lab in enumerate(labels):
        ece_info = expected_calibration_error(y_arr[:, j], p_arr[:, j], n_bins=n_bins)
        per_label[lab] = {
            "ece": ece_info["ece"],
            "mce": ece_info["mce"],
            "brier": brier_score(y_arr[:, j], p_arr[:, j]),
            "nll": nll_score(y_arr[:, j], p_arr[:, j]),
            "bins": ece_info,
        }
    macro = {
        metric: float(np.mean([per_label[lab][metric] for lab in labels]))
        for metric in ("ece", "mce", "brier", "nll")
    }
    return {"per_label": per_label, "macro": macro}


def plot_reliability_diagrams(
    y_true: np.ndarray,
    prob_variants: dict[str, np.ndarray],
    labels: list[str],
    out_path: str | Path,
    title: str = "Reliability Diagrams (Test Split)",
    n_bins: int = 10,
) -> Path:
    """Save 1xN reliability curves comparing calibration methods per label."""
    import matplotlib.pyplot as plt

    n_labels = len(labels)
    fig, axes = plt.subplots(1, n_labels, figsize=(5.2 * n_labels, 4.6), sharey=True)
    if n_labels == 1:
        axes = [axes]

    colors = ["#dc2626", "#f59e0b", "#16a34a", "#2563eb", "#7c3aed"]
    markers = ["o", "s", "^", "D", "v"]

    for j, lab in enumerate(labels):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Perfectly calibrated")
        for idx, (vname, probs) in enumerate(prob_variants.items()):
            info = expected_calibration_error(y_true[:, j], probs[:, j], n_bins=n_bins)
            xs = [c for c in info["bin_conf"] if c is not None]
            ys = [a for a in info["bin_acc"] if a is not None]
            ax.plot(
                xs,
                ys,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                lw=2.0,
                ms=5.5,
                label=f"{vname} (ECE={info['ece']:.3f})",
            )
        ax.set_title("ACL" if lab.lower() == "acl" else lab.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        if j == 0:
            ax.set_ylabel("Observed Positive Fraction", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1.01)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
