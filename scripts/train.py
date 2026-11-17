"""Train one encoder with the shared protocol.

  python scripts/train.py --model resnet50 --smoke --no-pretrained   # 2-minute pipeline check
  python scripts/train.py --model resnet50                           # full run
  python scripts/train.py --model resnet50 --resume                  # continue after a disconnect
"""
import argparse

from orthovision.config import load_config
from orthovision.engine import fit

cfg = load_config()
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=cfg["models"]["names"])
ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs")
ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--run-name", default=None)
a = ap.parse_args()
fit(cfg, a.model, a.run_name, a.smoke, not a.no_pretrained, a.resume, a.epochs)
