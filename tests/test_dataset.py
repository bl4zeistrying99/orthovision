import pytest

torch = pytest.importorskip("torch")

from orthovision.data.dataset import MRNetDataset, collate_exams  # noqa: E402


def test_dataset_and_collate(fake_mrnet):
    ds = MRNetDataset(fake_mrnet, "train", None, ["axial", "coronal", "sagittal"],
                      ["abnormal", "acl", "meniscus"], image_size=32)
    assert len(ds) == 6
    batch = collate_exams([ds[0], ds[1]])
    assert batch["y"].shape == (2, 3)
    for p in ["axial", "coronal", "sagittal"]:
        x, m = batch["series"][p], batch["mask"][p]
        assert x.shape[0] == 2 and x.shape[2:] == (3, 32, 32)
        assert m.shape == x.shape[:2] and m.any(dim=1).all()
