"""Check the MRNet folder layout, label files and a sample volume.  python scripts/verify_data.py"""
import sys

import numpy as np

from orthovision.config import load_config, resolve
from orthovision.data.labels import load_labels, pos_weights

cfg = load_config()
root = resolve(cfg, "data_root")
planes, labels = cfg["data"]["planes"], cfg["data"]["labels"]
ok = True
print(f"Data root: {root}")
if not root.exists():
    sys.exit("  [X] data_root not found. See data/README.md")

for folder, expected in [("train", 1130), ("valid", 120)]:
    try:
        df = load_labels(root, folder, labels)
    except FileNotFoundError as e:
        print(f"  [X] {e}")
        ok = False
        continue
    print(f"  {folder}: {len(df)} exams (expected {expected})  positives: "
          + ", ".join(f"{c}={int(df[c].sum())}" for c in labels))
    if folder == "train":
        print("  pos_weight (neg/pos):", {k: round(v, 2) for k, v in pos_weights(df).items()})
    missing = [
        (p, i) for p in planes for i in df.index if not (root / folder / p / f"{i:04d}.npy").exists()
    ]
    if missing:
        print(f"  [X] {len(missing)} missing .npy files, e.g. {missing[:3]}")
        ok = False
    else:
        v = np.load(root / folder / planes[0] / f"{df.index[0]:04d}.npy")
        print(f"  [OK] all volumes present; sample shape {v.shape} dtype {v.dtype}")
print("\nData OK" if ok else "\nData has problems")
sys.exit(0 if ok else 1)
