"""K-means colour-clustering baseline (Section 5.2).

Each test image is reshaped to (H*W, 3) and clustered with k-means++ (NumPy
implementation, no scikit dependency at inference time). The three resulting
clusters are sorted by mean luma and assigned background -> cytoplasm -> nucleus
from brightest to darkest.

Usage:
    python -m src.baselines.kmeans --config configs/kmeans.yaml \
                                   --report-json results/metrics/kmeans_test.json
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


def _kmeans_pp_init(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = pts.shape[0]
    centers = np.empty((k, pts.shape[1]), dtype=np.float64)
    centers[0] = pts[rng.integers(0, n)]
    closest_sq = np.full(n, np.inf)
    for i in range(1, k):
        diff = pts - centers[i - 1]
        d_sq = (diff * diff).sum(axis=1)
        closest_sq = np.minimum(closest_sq, d_sq)
        s = closest_sq.sum()
        probs = closest_sq / s if s > 0 else np.ones(n) / n
        centers[i] = pts[rng.choice(n, p=probs)]
    return centers


def _kmeans(pts: np.ndarray, k: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(pts, k, rng)
    labels = np.zeros(pts.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        # Assign.
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update.
        for c in range(k):
            sel = pts[labels == c]
            if sel.size:
                centers[c] = sel.mean(axis=0)
    return labels, centers


def segment_image(img_chw_uint8: np.ndarray, k: int, max_iter: int, seed: int,
                  subsample: int = 4) -> np.ndarray:
    """Cluster pixel colours and map clusters to {background, cytoplasm, nucleus}."""
    img = np.transpose(img_chw_uint8, (1, 2, 0)).astype(np.float64)
    h, w, _ = img.shape
    flat = img.reshape(-1, 3)
    # Speed: fit on a subsample, then assign to the full image.
    fit_pts = flat[::subsample] if subsample > 1 else flat
    _, centers = _kmeans(fit_pts, k=k, max_iter=max_iter, seed=seed)
    d = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d.argmin(axis=1).reshape(h, w)

    # Sort clusters by luma: brightest -> background, darkest -> nucleus.
    luma = (0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2])
    order = np.argsort(-luma)        # bright to dark
    target = {int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}
    out = np.zeros_like(labels, dtype=np.uint8)
    for src, dst in target.items():
        out[labels == src] = dst
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))

    test_ds = CytologyDataset(cfg["data"]["root"], split="test", cfg=cfg,
                              transform=build_eval_transform(cfg["data"]))
    print(f"[kmeans] test={len(test_ds)}")
    k = int(cfg["baseline"]["n_clusters"])
    max_iter = int(cfg["baseline"]["max_iter"])
    seed = int(cfg["experiment"]["seed"])

    preds, trues, pred_nc, true_nc = [], [], [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        img_uint8 = (np.clip(sample["image"].numpy(), -3, 3) * 64.0 + 128.0).clip(0, 255).astype(np.uint8)
        pred = segment_image(img_uint8, k=k, max_iter=max_iter, seed=seed)
        true = sample["mask"].numpy()
        preds.append(pred); trues.append(true)
        pred_nc.append(field_of_view_mean(pred))
        true_nc.append(field_of_view_mean(true))

    pred = np.stack(preds, axis=0)
    true = np.stack(trues, axis=0)
    m = M.compute(pred, true, pred_nc=pred_nc, true_nc=true_nc,
                  num_classes=NUM_CLASSES, nucleus_class=2)
    out = m.to_dict()
    out["method"] = "kmeans_pp"
    out["n_test_images"] = int(len(test_ds))

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
