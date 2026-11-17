"""MRNet exam-level Dataset. One item = one exam = up to 3 planes of variable slice count."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .labels import load_labels
from .preprocess import preprocess_series


def augment_series(x: torch.Tensor, max_shift_frac: float = 0.08, contrast: float = 0.1,
                   brightness: float = 0.1) -> torch.Tensor:
    """Light, anatomy-preserving augmentation on one series (S,3,H,W): one random translation
    and one intensity jitter shared by all slices. No flips: medial/lateral must stay meaningful
    for the later Grad-CAM region mapping."""
    _, _, h, w = x.shape
    m = int(max_shift_frac * h)
    if m > 0:
        dy, dx = torch.randint(-m, m + 1, (2,)).tolist()
        xp = F.pad(x, (m, m, m, m), mode="reflect")
        x = xp[:, :, m + dy : m + dy + h, m + dx : m + dx + w]
    sc = 1 + (torch.rand(1).item() * 2 - 1) * contrast
    br = (torch.rand(1).item() * 2 - 1) * brightness
    return (x * sc + br).contiguous()


class MRNetDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        folder: str,                 # 'train' or 'valid' (official folder names)
        exam_ids: list[int] | None,  # None -> every exam in the folder
        planes: list[str],
        labels: list[str],
        image_size: int = 224,
        train: bool = False,
        max_slices: int | None = None,
    ):
        self.root = Path(data_root)
        self.folder = folder
        self.planes = planes
        self.labels = labels
        self.size = image_size
        self.train = train
        self.max_slices = max_slices
        self.df = load_labels(self.root, folder, labels)
        if exam_ids is not None:
            self.df = self.df.loc[sorted(exam_ids)]
        self.ids = self.df.index.tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        eid = self.ids[i]
        series = {}
        for p in self.planes:
            vol = np.load(self.root / self.folder / p / f"{eid:04d}.npy")
            if self.train and self.max_slices and vol.shape[0] > self.max_slices:
                keep = torch.randperm(vol.shape[0])[: self.max_slices].sort().values.numpy()
                vol = vol[keep]
            x = torch.from_numpy(preprocess_series(vol, self.size))  # (S,3,H,W)
            series[p] = augment_series(x) if self.train else x
        y = torch.tensor(self.df.loc[eid, self.labels].values.astype("float32"))
        return {"id": eid, "series": series, "y": y}


def collate_exams(batch: list[dict]) -> dict:
    """Pad slice dimension per plane; return a boolean mask so max-pooling ignores padding."""
    out = {"id": [b["id"] for b in batch], "y": torch.stack([b["y"] for b in batch])}
    series, masks = {}, {}
    for p in batch[0]["series"]:
        smax = max(b["series"][p].shape[0] for b in batch)
        x = torch.zeros(len(batch), smax, *batch[0]["series"][p].shape[1:])
        m = torch.zeros(len(batch), smax, dtype=torch.bool)
        for j, b in enumerate(batch):
            s = b["series"][p].shape[0]
            x[j, :s] = b["series"][p]
            m[j, :s] = True
        series[p], masks[p] = x, m
    out["series"], out["mask"] = series, masks
    return out
