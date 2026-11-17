import numpy as np

from orthovision.eval.metrics import auc_metrics


def test_perfect_and_single_class():
    y = np.array([[0, 1], [1, 1], [0, 1], [1, 1]])
    p = np.array([[0.1, 0.5], [0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])
    m = auc_metrics(y, p, ["a", "b"])
    assert m["auc_a"] == 1.0 and np.isnan(m["auc_b"]) and m["auc_macro"] == 1.0
