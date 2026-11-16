"""Dataset statistics + sample slices -> outputs/figures/.   python scripts/eda.py"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orthovision.config import REPO_ROOT, load_config, resolve
from orthovision.data.labels import load_labels

cfg = load_config()
root, planes, labels = resolve(cfg, "data_root"), cfg["data"]["planes"], cfg["data"]["labels"]
figs = resolve(cfg, "outputs") / "figures"
figs.mkdir(parents=True, exist_ok=True)

df = load_labels(root, "train", labels)
print("Positive rate (train):")
for c in labels:
    print(f"  {c:<10} {df[c].mean():.1%}  ({int(df[c].sum())}/{len(df)})")
print("Label co-occurrence (rows=has label):")
print(df.corr().round(2).to_string())

# slice counts + intensity range on a sample
ids = df.index[:: max(1, len(df) // 60)].tolist()
counts = {p: [] for p in planes}
for p in planes:
    for i in ids:
        counts[p].append(np.load(root / "train" / p / f"{i:04d}.npy", mmap_mode="r").shape[0])
    print(f"slices/{p}: min={min(counts[p])} median={int(np.median(counts[p]))} max={max(counts[p])}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].bar(labels, [df[c].mean() for c in labels])
ax[0].set_title("Positive rate (train)")
for p in planes:
    ax[1].hist(counts[p], alpha=0.6, label=p)
ax[1].set_title("Slices per series")
ax[1].legend()
fig.tight_layout()
fig.savefig(figs / "eda_summary.png", dpi=120)

# one exam per row: mid-slice of each plane; pick a healthy and an ACL-positive exam
picks = [("healthy", df[(df.abnormal == 0)].index[0]), ("ACL tear", df[df.acl == 1].index[0])]
fig, ax = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
for r, (name, i) in enumerate(picks):
    for c, p in enumerate(planes):
        v = np.load(root / "train" / p / f"{i:04d}.npy")
        ax[r, c].imshow(v[v.shape[0] // 2], cmap="gray")
        ax[r, c].set_title(f"{name} | exam {i} | {p}", fontsize=8)
        ax[r, c].axis("off")
fig.tight_layout()
fig.savefig(figs / "eda_samples.png", dpi=120)
print(f"Saved figures to {figs}")
