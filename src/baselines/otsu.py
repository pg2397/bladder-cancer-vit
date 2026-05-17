"""Multi-level Otsu thresholding baseline (Section 5.2).

The image is converted to greyscale, two thresholds are searched exhaustively
on the 256-bin histogram, and the three intensity bands are mapped to
background / cytoplasm / nucleus by area heuristic (smallest dark band is
the nucleus). No training is needed -- it is run directly on the test split.

Usage:
    python -m src.baselines.otsu --config configs/otsu.yaml \
                                 --report-json results/metrics/otsu_test.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..augment import build_eval_transform
from ..config import load_config
from ..dataset import CytologyDataset, NUM_CLASSES
from ..nc_ratio import field_of_view_mean
from ..seed import seed_everything


def _otsu_two_thresholds(gray: np.ndarray) -> tuple[int, int]:
    """Exhaustive 2-threshold Otsu search on a single greyscale image."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    p = hist.astype(np.float64) / max(total, 1)
    intensities = np.arange(256, dtype=np.float64)

    best_var, best = -1.0, (85, 170)
    # Skip degenerate cuts at the extreme bins.
    for t1 in range(1, 254):
        for t2 in range(t1 + 1, 255):
            w0 = p[:t1].sum()
            w1 = p[t1:t2].sum()
            w2 = p[t2:].sum()
            if w0 == 0 or w1 == 0 or w2 == 0:
                continue
            m0 = (intensities[:t1] * p[:t1]).sum() / w0
            m1 = (intensities[t1:t2] * p[t1:t2]).sum() / w1
            m2 = (intensities[t2:] * p[t2:]).sum() / w2
            mg = w0 * m0 + w1 * m1 + w2 * m2
            var = (w0 * (m0 - mg) ** 2
                   + w1 * (m1 - mg) ** 2
                   + w2 * (m2 - mg) ** 2)
            if var > best_var:
                best_var, best = var, (t1, t2)
    return best


def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)


def segment_image(img_chw_uint8: np.ndarray) -> np.ndarray:
    """Return a (H, W) mask with values {0, 1, 2}."""
    img = np.transpose(img_chw_uint8, (1, 2, 0))  # CHW -> HWC
    gray = _rgb_to_gray(img)
    t1, t2 = _otsu_two_thresholds(gray)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray <= t1] = 2          # darkest band = nucleus
    mask[(gray > t1) & (gray <= t2)] = 1   # mid band = cytoplasm
    mask[gray > t2] = 0           # brightest band = background
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))

    test_ds = CytologyDataset(cfg["data"]["root"], split="test", cfg=cfg,
                              transform=build_eval_transform(cfg["data"]))
    print(f"[otsu] test={len(test_ds)}")
    preds, trues, pred_nc, true_nc = [], [], [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        # Undo ImageNet normalisation enough to recover an 8-bit image.
        img_uint8 = (np.clip(sample["image"].numpy(), -3, 3) * 64.0 + 128.0).clip(0, 255).astype(np.uint8)
        pred = segment_image(img_uint8)
        true = sample["mask"].numpy()
        preds.append(pred); trues.append(true)
        pred_nc.append(field_of_view_mean(pred))
        true_nc.append(field_of_view_mean(true))

    pred = np.stack(preds, axis=0)
    true = np.stack(trues, axis=0)
    m = M.compute(pred, true, pred_nc=pred_nc, true_nc=true_nc,
                  num_classes=NUM_CLASSES, nucleus_class=2)
    out = m.to_dict()
    out["method"] = "otsu_multilevel"
    out["n_test_images"] = int(len(test_ds))

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
