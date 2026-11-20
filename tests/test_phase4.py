"""Unit tests for Phase 4: calibration, DeLong test, ensembles, and MC Dropout utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from orthovision.eval.calibration import (
    PlattScaler,
    TemperatureScaler,
    brier_score,
    calibration_report,
    expected_calibration_error,
)
from orthovision.eval.delong import (
    delong_roc_test,
    stratified_bootstrap_ci,
    threshold_metrics,
    youden_optimal_threshold,
)
from orthovision.eval.ensemble import average_ensemble, save_predictions_csv
from orthovision.uncertainty.mc_dropout import (
    enable_head_dropout,
    selective_referral_curve,
    simulate_head_mc_from_probs,
    summarize_mc_samples,
)


def test_brier_and_ece_exact_cases():
    y = np.array([0, 0, 1, 1], dtype=float)
    p_exact = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    assert brier_score(y, p_exact) == 0.0
    rep = expected_calibration_error(y, p_exact, n_bins=10)
    assert rep["ece"] == 0.0
    assert rep["mce"] == 0.0
    assert sum(rep["bin_counts"]) == 4

    p_const = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    assert abs(brier_score(y, p_const) - 0.25) < 1e-12
    assert abs(expected_calibration_error(y, p_const, n_bins=10)["ece"] - 0.0) < 1e-12


def test_temperature_and_platt_preserve_auc_and_fix_shift():
    rng = np.random.default_rng(123)
    labels = ["abnormal", "acl", "meniscus"]

    def _make_split(n: int):
        y_list, p_list = [], []
        for j, shift in enumerate([-1.2, 1.8, 0.9]):
            z_true = rng.normal(0.0, 1.2, size=n)
            p_true = expit(z_true)
            y_col = rng.binomial(1, p_true)
            # Simulate pos_weight intercept shift + overconfidence (2.2x slope)
            p_obs = expit(2.2 * z_true + shift)
            y_list.append(y_col)
            p_list.append(p_obs)
        return np.column_stack(y_list), np.column_stack(p_list)

    y_val, p_val = _make_split(400)
    y_test, p_test = _make_split(300)

    temp = TemperatureScaler(labels).fit(y_val, p_val, labels)
    platt = PlattScaler(labels).fit(y_val, p_val, labels)

    p_temp = temp.transform(p_test)
    p_platt = platt.transform(p_test)

    for j, lab in enumerate(labels):
        auc_raw = roc_auc_score(y_test[:, j], p_test[:, j])
        auc_t = roc_auc_score(y_test[:, j], p_temp[:, j])
        auc_p = roc_auc_score(y_test[:, j], p_platt[:, j])
        assert abs(auc_raw - auc_t) < 1e-6
        assert abs(auc_raw - auc_p) < 1e-6
        assert temp.temperatures_[lab] > 1.0
        assert platt.params_[lab]["a"] > 0.0

    rep_raw = calibration_report(y_test, p_test, labels)
    rep_platt = calibration_report(y_test, p_platt, labels)
    assert rep_platt["macro"]["ece"] < rep_raw["macro"]["ece"]
    assert rep_platt["macro"]["nll"] < rep_raw["macro"]["nll"]
    assert rep_platt["macro"]["brier"] < rep_raw["macro"]["brier"]


def test_delong_roc_test_symmetries_and_significance():
    rng = np.random.default_rng(7)
    y = np.concatenate([np.zeros(80), np.ones(80)])
    p_weak = expit(rng.normal(0.0, 1.0, size=160) + 0.2 * y)
    p_strong = expit(rng.normal(0.0, 0.7, size=160) + 2.2 * y)

    same = delong_roc_test(y, p_strong, p_strong)
    assert same["diff"] == 0.0
    assert same["z"] == 0.0
    assert same["p_value"] == 1.0

    res_ab = delong_roc_test(y, p_strong, p_weak)
    res_ba = delong_roc_test(y, p_weak, p_strong)
    assert res_ab["diff"] > 0.1
    assert res_ab["p_value"] < 0.01
    assert abs(res_ab["diff"] + res_ba["diff"]) < 1e-12
    assert abs(res_ab["z"] + res_ba["z"]) < 1e-12
    assert abs(res_ab["p_value"] - res_ba["p_value"]) < 1e-12


def test_stratified_bootstrap_ci_and_threshold_metrics():
    rng = np.random.default_rng(42)
    y = np.array([0] * 50 + [1] * 50)
    p = np.clip(0.2 + 0.6 * y + rng.normal(0.0, 0.15, size=100), 0.01, 0.99)
    auc = roc_auc_score(y, p)
    lo, hi = stratified_bootstrap_ci(y, p, n_boot=400, seed=42)
    assert 0.0 <= lo <= auc <= hi <= 1.0

    thr = youden_optimal_threshold(y, p)
    assert 0.0 <= thr <= 1.0
    tm = threshold_metrics(y, p, threshold=thr)
    assert tm["tp"] + tm["fn"] == 50
    assert tm["tn"] + tm["fp"] == 50
    assert tm["sensitivity"] > 0.75
    assert tm["specificity"] > 0.75


def test_average_ensemble_and_mismatch_guard():
    labels = ["abnormal", "acl", "meniscus"]
    ids = np.array([101, 102, 103])
    ys = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0]])
    p1 = np.array([[0.8, 0.2, 0.7], [0.1, 0.1, 0.2], [0.9, 0.8, 0.3]])
    p2 = np.array([[0.6, 0.4, 0.5], [0.3, 0.1, 0.4], [0.7, 0.6, 0.5]])

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        f1 = save_predictions_csv(d / "m1.csv", ids, p1, ys, labels)
        f2 = save_predictions_csv(d / "m2.csv", ids, p2, ys, labels)

        e_ids, e_y, e_p = average_ensemble([f1, f2], labels)
        assert np.array_equal(e_ids, ids)
        assert np.array_equal(e_y, ys)
        assert np.allclose(e_p, 0.5 * (p1 + p2), atol=1e-6)

        _, _, w_p = average_ensemble([f1, f2], labels, weights=[3.0, 1.0])
        assert np.allclose(w_p, 0.75 * p1 + 0.25 * p2, atol=1e-6)

        f_bad = save_predictions_csv(d / "bad.csv", np.array([101, 102, 999]), p2, ys, labels)
        try:
            average_ensemble([f1, f_bad], labels)
            raise AssertionError("Expected ValueError for mismatched IDs")
        except ValueError:
            pass


def test_mc_summary_and_selective_referral():
    y_true = np.array([[1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=np.int64)
    p_base = np.array([[0.85, 0.10, 0.70], [0.15, 0.20, 0.15], [0.90, 0.85, 0.30], [0.48, 0.52, 0.51]])

    mc_zero = simulate_head_mc_from_probs(p_base, dropout_p=0.0, n_passes=10, seed=42)
    sum_zero = summarize_mc_samples(mc_zero, y_true=y_true)
    assert np.allclose(sum_zero["p_std"], 0.0)
    assert np.allclose(sum_zero["mutual_information"], 0.0)

    mc_30 = simulate_head_mc_from_probs(p_base, dropout_p=0.3, n_passes=30, seed=42)
    sum_30 = summarize_mc_samples(mc_30, y_true=y_true)
    assert sum_30["p_mean"].shape == (4, 3)
    assert np.all(sum_30["p_std"] > 0.0)
    assert np.all(sum_30["mutual_information"] >= 0.0)

    # Uncertain predictions have lower accuracy than confident ones -> referral improves accuracy
    y_ref = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    p_ref = np.array([0.05, 0.10, 0.08, 0.92, 0.88, 0.95, 0.91, 0.12, 0.45, 0.55])
    u_ref = np.array([0.02, 0.03, 0.01, 0.02, 0.04, 0.01, 0.03, 0.02, 0.25, 0.30])
    curve = selective_referral_curve(y_ref, p_ref, u_ref, fractions=(0.0, 0.2))
    assert curve[0]["accuracy"] == 0.8
    assert curve[1]["accuracy"] == 1.0
    assert curve[1]["kept_count"] == 8


def test_enable_head_dropout_only_keeps_head_dropout_training():
    try:
        import torch.nn as nn
    except ImportError:
        try:
            import pytest

            pytest.skip("torch not installed in local CPU test environment")
        except ImportError:
            return

    class _DummyExam(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(4, 3)

    m = _DummyExam()
    m.train()
    enable_head_dropout(m)
    assert m.training is False
    assert m.backbone[1].training is False
    assert m.dropout.training is True
    assert m.head.training is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except BaseException as exc:
                if type(exc).__name__ == "Skipped":
                    print(f"[SKIP] {name}: {exc}")
                else:
                    raise
    print("All Phase 4 unit tests passed.")
