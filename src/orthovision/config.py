"""Config loading: configs/default.yaml, deep-merged with optional configs/local.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_dir = REPO_ROOT / "configs"
    with open(path or cfg_dir / "default.yaml") as f:
        cfg = yaml.safe_load(f)
    local = cfg_dir / "local.yaml"
    if path is None and local.exists():
        with open(local) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a cfg['paths'][key] entry against the repo root (absolute paths pass through)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p
