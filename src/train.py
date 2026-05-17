"""Training entry point.

Usage:
    python -m src.train --config configs/vit.yaml
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from .augment import build_eval_transform, build_train_transform
from .config import load_config
from .dataset import CytologyDataset, NUM_CLASSES, compute_class_weights
from .model import build_model
from .seed import seed_everything, worker_init_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]),
                    deterministic=bool(cfg["experiment"].get("deterministic", True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    train_tf = build_train_transform(cfg["data"])
    eval_tf = build_eval_transform(cfg["data"])
    train_ds = CytologyDataset(cfg["data"]["root"], split="train", cfg=cfg, transform=train_tf)
    val_ds = CytologyDataset(cfg["data"]["root"], split="val", cfg=cfg, transform=eval_tf)
    print(f"[train] train={len(train_ds)} val={len(val_ds)}")

    tr_cfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg.get("val_batch_size", tr_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(tr_cfg.get("num_workers", 4)),
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"], num_classes=NUM_CLASSES).to(device)

    # Class weights from config if present; else compute from the data.
    cw = cfg.get("loss", {}).get("class_weights")
    weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw \
        else compute_class_weights(train_loader, NUM_CLASSES).to(device)
    print(f"[train] class_weights={weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # Differential learning rate (see Section 4.9).
    optim_cfg = cfg["optim"]
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optim = torch.optim.AdamW(
        [{"params": enc_params, "lr": float(optim_cfg["encoder_lr"])},
         {"params": dec_params, "lr": float(optim_cfg["decoder_lr"])}],
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-4)),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=int(optim_cfg.get("scheduler_T_max", tr_cfg["epochs"]))
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(tr_cfg.get("amp", True)))
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = int(tr_cfg.get("checkpoint_top_k", 3))
    history: list[dict] = []
    best: list[tuple[float, Path]] = []  # (val_macro_f1, path)

    for epoch in range(1, int(tr_cfg["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(img)
            # Force fp32 loss to avoid the NaN failure described in Section 4.12.
            loss = criterion(logits.float(), mask)
            scaler.scale(loss).backward()
            if optim_cfg.get("grad_clip"):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_cfg["grad_clip"]))
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * img.size(0)
        sched.step()

        val_m = _evaluate(model, val_loader, device)
        avg_loss = running / max(len(train_ds), 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_m.to_dict()})
        print(f"[epoch {epoch:>3d}] loss={avg_loss:.4f}  val_macroF1={val_m.macro_f1:.4f}")

        ckpt = out_dir / f"epoch_{epoch:03d}.pth"
        torch.save({"model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_m.macro_f1}, ckpt)
        best.append((val_m.macro_f1, ckpt))
        best.sort(key=lambda x: -x[0])
        # Prune to top-k by val_macro_f1.
        for _, p in best[top_k:]:
            if p.exists():
                p.unlink()
        best = best[:top_k]

    # Save the SWA-style top-k ensemble weights (see Section 4.12).
    weights_out = Path(cfg["paths"]["weights_out"])
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    if best:
        avg_state = _average_checkpoints([p for _, p in best],
                                         scores=[s for s, _ in best])
        torch.save(avg_state, weights_out)
        print(f"[train] saved ensembled weights -> {weights_out}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# --------------------------------------------------------------- helpers
def _evaluate(model, loader, device) -> M.SegmentationMetrics:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].numpy()
            logits = model(img).cpu().numpy()
            preds.append(np.argmax(logits, axis=1))
            trues.append(mask)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    return M.compute(pred, true, num_classes=NUM_CLASSES, nucleus_class=2)


def _average_checkpoints(paths, scores):
    """Weighted average of state dicts -- weights are the per-checkpoint scores."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
    state_dicts = [torch.load(p, map_location="cpu")["model"] for p in paths]
    avg = {}
    for k in state_dicts[0].keys():
        if state_dicts[0][k].dtype.is_floating_point:
            avg[k] = sum(float(w) * sd[k].float() for w, sd in zip(weights, state_dicts))
            avg[k] = avg[k].to(state_dicts[0][k].dtype)
        else:
            avg[k] = state_dicts[0][k]
    return avg


if __name__ == "__main__":
    main()
