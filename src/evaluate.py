"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python -m src.evaluate --config configs/vit.yaml \
                           --weights results/weights/vit_seg_best.pth \
                           --report-json results/metrics/vit_test.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .augment import build_eval_transform
from .config import load_config
from .dataset import CytologyDataset, NUM_CLASSES
from .model import build_model
from .nc_ratio import field_of_view_mean
from .seed import seed_everything, worker_init_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]),
                    deterministic=bool(cfg["experiment"].get("deterministic", True)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = CytologyDataset(cfg["data"]["root"], split="test", cfg=cfg,
                              transform=build_eval_transform(cfg["data"]))
    print(f"[eval] test={len(test_ds)} device={device}")
    loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                        num_workers=2, worker_init_fn=worker_init_fn,
                        pin_memory=device.type == "cuda")

    model = build_model(cfg["model"], num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.weights, map_location=device)
    state = state.get("model", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    preds, trues, pred_nc, true_nc = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].numpy()
            logits = model(img).cpu().numpy()
            p = np.argmax(logits, axis=1)
            preds.append(p); trues.append(mask)
            for i in range(p.shape[0]):
                pred_nc.append(field_of_view_mean(p[i]))
                true_nc.append(field_of_view_mean(mask[i]))

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    m = M.compute(pred, true, pred_nc=pred_nc, true_nc=true_nc,
                  num_classes=NUM_CLASSES, nucleus_class=2)
    out = m.to_dict()
    out["n_test_images"] = int(len(test_ds))
    out["weights"] = str(args.weights)

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
