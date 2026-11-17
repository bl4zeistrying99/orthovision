"""Training / evaluation engine shared by ALL encoders (identical-protocol comparison)."""
from __future__ import annotations

import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import REPO_ROOT, resolve
from .data.dataset import MRNetDataset, collate_exams
from .data.labels import pos_weights
from .eval.metrics import auc_metrics
from .models.exam_model import build_model
from .utils.seed import set_seed

SPLIT_FILE = REPO_ROOT / "splits" / "split_v1.json"


def _to_device(batch, device):
    series = {p: x.to(device, non_blocking=True) for p, x in batch["series"].items()}
    mask = {p: m.to(device, non_blocking=True) for p, m in batch["mask"].items()}
    return series, mask, batch["y"].to(device)


@torch.no_grad()
def predict(model, loader, device, amp: bool = True):
    """Return (ids, probs, y_true) as numpy arrays. Dropout off, probabilities = sigmoid(logit)."""
    model.eval()
    ids, probs, ys = [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        series, mask, _ = _to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(series, mask)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(batch["y"].numpy())
        ids += batch["id"]
    return np.array(ids), np.concatenate(probs), np.concatenate(ys)


def _save_predictions(path, ids, probs, ys, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [f"y_{l}" for l in labels] + [f"p_{l}" for l in labels])
        for i, y, p in zip(ids, ys, probs):
            w.writerow([int(i)] + [int(v) for v in y] + [f"{v:.6f}" for v in p])


def fit(cfg, model_name, run_name=None, smoke=False, pretrained=True, resume=False, epochs=None):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, labels, planes = cfg["train"], cfg["data"]["labels"], cfg["data"]["planes"]
    run = run_name or f"{model_name}{'_smoke' if smoke else ''}_seed{cfg['seed']}"
    out = resolve(cfg, "outputs")
    ckpt_dir, log_dir, pred_dir = out / "checkpoints", out / "logs" / run, out / "predictions"
    for d in (ckpt_dir, log_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    split = json.loads(SPLIT_FILE.read_text())
    root = resolve(cfg, "data_root")
    n_cap = {"train": 24, "val": 12, "test": 12}

    def make(name, train):
        ids = split[name]["ids"][: n_cap[name]] if smoke else split[name]["ids"]
        return MRNetDataset(
            root, split[name]["folder"], ids, planes, labels, cfg["data"]["image_size"],
            train=train, max_slices=(8 if smoke else t["max_slices_train"]) if train else None,
        )

    train_ds, val_ds, test_ds = make("train", True), make("val", False), make("test", False)
    g = torch.Generator().manual_seed(cfg["seed"])
    nw = cfg["data"]["num_workers"]
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, collate_fn=collate_exams,
                          num_workers=nw, pin_memory=True, generator=g)
    mk_eval = lambda ds: DataLoader(ds, t["eval_batch_size"], shuffle=False,  # noqa: E731
                                    collate_fn=collate_exams, num_workers=nw, pin_memory=True)
    val_dl, test_dl = mk_eval(val_ds), mk_eval(test_ds)

    model = build_model(model_name, cfg, pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params) / 1e6
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=t["lr_patience"])
    use_amp = bool(t["amp"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if t["use_pos_weight"]:
        pw = pos_weights(train_ds.df)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw[l] for l in labels]).to(device))
        print("pos_weight:", {k: round(v, 2) for k, v in pw.items()})
    else:
        crit = nn.BCEWithLogitsLoss()

    print(f"[{run}] device={device} params={n_params:.1f}M train/val/test="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} amp={use_amp}")

    best_path, last_path = ckpt_dir / f"{run}_best.pt", ckpt_dir / f"{run}_last.pt"
    start, best, bad = 0, -1.0, 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed from epoch {start}")

    log_path = log_dir / "log.csv"
    new_log = not (resume and log_path.exists())
    n_epochs = epochs or (2 if smoke else t["epochs"])
    accum = 2 if smoke else t["accum_steps"]

    for epoch in range(start, n_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, total = time.time(), 0.0
        for step, batch in enumerate(tqdm(train_dl, desc=f"epoch {epoch}", leave=False)):
            series, mask, y = _to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(series, mask)
            loss = crit(logits.float(), y)
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_dl):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, t["grad_clip"])
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item()

        _, vp, vy = predict(model, val_dl, device, use_amp)
        m = auc_metrics(vy, vp, labels)
        score = 0.0 if np.isnan(m["auc_macro"]) else m["auc_macro"]
        sched.step(score)
        row = {"epoch": epoch, "train_loss": total / len(train_dl), "lr": opt.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in m.items()}}
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new_log:
                w.writeheader(); new_log = False
            w.writerow(row)
        print(row)

        if score > best:
            best, bad = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": m, "cfg": cfg,
                        "model_name": model_name}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad}, last_path)
        if bad >= t["early_stop_patience"]:
            print(f"early stop at epoch {epoch}")
            break

    # final: best checkpoint -> save val + test predictions (needed for ensemble/DeLong/calibration)
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    res = {}
    for name, dl in (("val", val_dl), ("test", test_dl)):
        ids, p, y = predict(model, dl, device, use_amp)
        _save_predictions(pred_dir / f"{run}_{name}.csv", ids, p, y, labels)
        res[name] = auc_metrics(y, p, labels)
    (log_dir / "final_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res
