import numpy as np

from orthovision.data.labels import load_labels, make_splits, pos_weights
from orthovision.data.preprocess import normalize_intensity, preprocess_series


def test_normalize_range():
    x = normalize_intensity(np.random.randint(0, 255, (5, 32, 32)))
    assert x.min() == 0.0 and x.max() == 1.0


def test_constant_volume_does_not_divide_by_zero():
    assert normalize_intensity(np.full((3, 8, 8), 7)).sum() == 0


def test_preprocess_shape():
    out = preprocess_series(np.random.rand(7, 256, 256), size=64)
    assert out.shape == (7, 3, 64, 64) and out.dtype == np.float32


def test_labels_and_splits(fake_mrnet):
    df = load_labels(fake_mrnet, "train", ["abnormal", "acl", "meniscus"])
    assert len(df) == 6 and set(df.columns) == {"abnormal", "acl", "meniscus"}
    tr, va = make_splits(df.index.tolist(), 0.34, seed=1)
    assert not set(tr) & set(va) and len(tr) + len(va) == 6   # no leakage
    assert make_splits(df.index.tolist(), 0.34, seed=1) == (tr, va)  # deterministic
    assert all(v >= 0 for v in pos_weights(df).values())
