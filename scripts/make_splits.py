"""Create the frozen exam-level split. Run ONCE, commit splits/split_v1.json, never regenerate.

  python scripts/make_splits.py
"""
import json
import sys

from orthovision.config import REPO_ROOT, load_config, resolve
from orthovision.data.labels import load_labels, make_splits

cfg = load_config()
root = resolve(cfg, "data_root")
labels = cfg["data"]["labels"]
out = REPO_ROOT / "splits" / "split_v1.json"
if out.exists():
    sys.exit(f"{out} already exists. The split is frozen; delete it only if the whole team agrees.")

train_df = load_labels(root, "train", labels)
test_df = load_labels(root, "valid", labels)
tr, va = make_splits(train_df.index.tolist(), cfg["data"]["val_fraction"], cfg["seed"])

split = {
    "seed": cfg["seed"],
    "val_fraction": cfg["data"]["val_fraction"],
    "train": {"folder": "train", "ids": tr},
    "val": {"folder": "train", "ids": va},
    "test": {"folder": "valid", "ids": sorted(test_df.index.tolist())},
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(split))
print(f"Wrote {out}")
for name in ["train", "val", "test"]:
    df = test_df if name == "test" else train_df
    sub = df.loc[split[name]["ids"]]
    print(f"  {name:<5} n={len(sub):<5} " + "  ".join(f"{c}={sub[c].mean():.1%}" for c in labels))
