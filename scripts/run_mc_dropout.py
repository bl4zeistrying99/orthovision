"""Run 30-pass head-only MC Dropout inference on GPU/Kaggle from a saved `*_best.pt` checkpoint.

Usage (on Kaggle or any machine with MRNet data + PyTorch + trained checkpoint):
  TQDM_DISABLE=1 python scripts/run_mc_dropout.py --model resnet50 --passes 30
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from orthovision.config import REPO_ROOT, load_config, resolve
from orthovision.data.dataset import MRNetDataset, collate_exams
from orthovision.models.exam_model import build_model
from orthovision.uncertainty.mc_dropout import mc_predict

SPLIT_FILE = REPO_ROOT / "splits" / "split_v1.json"


def _save_mc_csv(path: Path, summary: dict, labels: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["id"]
        + [f"y_{lab}" for lab in labels]
        + [f"p_mean_{lab}" for lab in labels]
        + [f"p_std_{lab}" for lab in labels]
        + [f"entropy_{lab}" for lab in labels]
        + [f"mi_{lab}" for lab in labels]
    )
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(summary["id"])):
            row = [int(summary["id"][i])]
            row += [int(v) for v in summary["y_true"][i]]
            for arr_key in ("p_mean", "p_std", "predictive_entropy", "mutual_information"):
                row += [f"{float(v):.6f}" for v in summary[arr_key][i]]
            w.writerow(row)
    return path


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Run T-pass MC Dropout uncertainty estimation.")
    ap.add_argument("--model", required=True, choices=cfg["models"]["names"])
    ap.add_argument("--ckpt", type=Path, default=None, help="Override path to *_best.pt")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--passes", type=int, default=cfg["uncertainty"]["mc_passes"])
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = ap.parse_args()

    out = resolve(cfg, "outputs")
    run = args.run_name or f"{args.model}_seed{cfg['seed']}"
    ckpt_path = args.ckpt or (out / "checkpoints" / f"{run}_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, cfg, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    split_data = json.loads(SPLIT_FILE.read_text())
    root = resolve(cfg, "data_root")
    labels = cfg["data"]["labels"]
    planes = cfg["data"]["planes"]
    splits = ["val", "test"] if args.split == "both" else [args.split]

    for sp in splits:
        ds = MRNetDataset(
            root,
            split_data[sp]["folder"],
            split_data[sp]["ids"],
            planes,
            labels,
            cfg["data"]["image_size"],
            train=False,
        )
        dl = DataLoader(
            ds,
            batch_size=cfg["train"]["eval_batch_size"],
            shuffle=False,
            collate_fn=collate_exams,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
        summary = mc_predict(
            model,
            dl,
            device=device,
            n_passes=args.passes,
            seed=cfg["seed"],
            amp=bool(cfg["train"]["amp"]),
        )
        out_csv = out / "predictions" / f"{run}_mc{args.passes}_{sp}.csv"
        _save_mc_csv(out_csv, summary, labels)
        print(
            f"[{run} | {sp}] saved {out_csv.name} | mean p_std="
            f"{summary['p_std'].mean(axis=0).round(4).tolist()}"
        )


if __name__ == "__main__":
    main()
