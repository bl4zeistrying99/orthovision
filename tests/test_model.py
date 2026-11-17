import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from orthovision.data.dataset import MRNetDataset, collate_exams  # noqa: E402
from orthovision.models.exam_model import ExamModel, masked_pool  # noqa: E402

PL = ["axial", "coronal", "sagittal"]


def test_masked_max_ignores_padding():
    f = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [99.0, 99.0]]])
    m = torch.tensor([[True, True, False]])
    assert masked_pool(f, m, "max").tolist() == [[3.0, 2.0]]
    assert masked_pool(f, m, "mean").tolist() == [[2.0, 1.0]]


def _tiny():
    bb = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(4))
    return ExamModel(bb, 4, PL, 3, slice_chunk=3, grad_checkpoint=True)


def test_forward_backward_shapes():
    model = _tiny()
    model.train()
    series = {p: torch.randn(2, 5, 3, 16, 16) for p in PL}
    mask = {p: torch.ones(2, 5, dtype=torch.bool) for p in PL}
    mask["axial"][1, 3:] = False
    out = model(series, mask)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert model.head.weight.grad is not None


def test_train_dataset_subsamples_slices(fake_mrnet):
    ds = MRNetDataset(fake_mrnet, "train", None, PL, ["abnormal", "acl", "meniscus"], 32,
                      train=True, max_slices=3)
    b = collate_exams([ds[0], ds[1]])
    assert all(b["series"][p].shape[1] == 3 for p in PL)
