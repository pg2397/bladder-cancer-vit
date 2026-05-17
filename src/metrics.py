"""Evaluation metrics: pixel accuracy, macro / per-class F1, Spearman r, MAE.

All five are described in Section 5.1 of the report.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


@dataclass
class SegmentationMetrics:
    pixel_accuracy: float
    macro_f1: float
    f1_per_class: list[float]
    nucleus_f1: float
    spearman_r: float
    mae: float

    def to_dict(self) -> dict:
        return asdict(self)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def per_class_f1(pred: np.ndarray, target: np.ndarray, num_classes: int) -> list[float]:
    out = []
    for c in range(num_classes):
        tp = float(((pred == c) & (target == c)).sum())
        fp = float(((pred == c) & (target != c)).sum())
        fn = float(((pred != c) & (target == c)).sum())
        denom = 2 * tp + fp + fn
        out.append(2 * tp / denom if denom > 0 else 0.0)
    return out


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation without bringing scipy in for one function."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def compute(
    pred_masks: np.ndarray,
    true_masks: np.ndarray,
    pred_nc: Sequence[float] | None = None,
    true_nc: Sequence[float] | None = None,
    num_classes: int = 3,
    nucleus_class: int = 2,
) -> SegmentationMetrics:
    pa = pixel_accuracy(pred_masks, true_masks)
    f1s = per_class_f1(pred_masks, true_masks, num_classes)
    macro = float(np.mean(f1s))
    nucleus = f1s[nucleus_class]
    if pred_nc is not None and true_nc is not None and len(pred_nc):
        sp = spearman_rho(pred_nc, true_nc)
        mae = float(np.mean(np.abs(np.asarray(pred_nc) - np.asarray(true_nc))))
    else:
        sp, mae = 0.0, 0.0
    return SegmentationMetrics(
        pixel_accuracy=pa,
        macro_f1=macro,
        f1_per_class=f1s,
        nucleus_f1=nucleus,
        spearman_r=sp,
        mae=mae,
    )
