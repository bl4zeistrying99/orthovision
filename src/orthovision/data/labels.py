"""MRNet label files + exam-level train/val/test splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_FILES = {"abnormal": "abnormal", "acl": "acl", "meniscus": "meniscus"}


def load_labels(data_root: Path, split: str, labels: list[str]) -> pd.DataFrame:
    """Return DataFrame indexed by exam id (int) with one 0/1 column per label.

    MRNet CSVs are headerless: `<exam_id>,<0|1>`, named e.g. train-acl.csv.
    """
    frames = []
    for lab in labels:
        f = Path(data_root) / f"{split}-{LABEL_FILES[lab]}.csv"
        df = pd.read_csv(f, header=None, names=["id", lab]).set_index("id")
        frames.append(df)
    out = pd.concat(frames, axis=1)
    out.index = out.index.astype(int)
    return out.sort_index()


def make_splits(
    train_ids: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Exam-level random split of the official train set into train / val.

    NOTE: MRNet exams are one-per-patient-visit; splitting by exam id is the finest unit
    the public data allows. We stratify on 'abnormal' elsewhere if needed.
    """
    ids = np.array(sorted(train_ids))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_fraction))
    return sorted(ids[n_val:].tolist()), sorted(ids[:n_val].tolist())


def pos_weights(df: pd.DataFrame) -> dict[str, float]:
    """neg/pos ratio per label for the weighted BCE in synopsis Eq. 4.1."""
    return {c: float((df[c] == 0).sum() / max((df[c] == 1).sum(), 1)) for c in df.columns}
