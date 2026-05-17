"""Per-cell N/C ratio computation from a 3-class segmentation mask.

The N/C ratio for a cell is

    N / (N + C)

where N is the nucleus pixel count and C is the cytoplasm pixel count
attached to the same connected cell region. See Section 4.10 of the report
for the full procedure.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Class indices, kept in sync with src.dataset.
BACKGROUND, CYTOPLASM, NUCLEUS = 0, 1, 2


@dataclass
class CellMeasurement:
    cell_id: int
    nucleus_area: int
    cytoplasm_area: int
    nc_ratio: float


def _label_connected(mask: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    """Tiny flood-fill connected-components labeller (no SciPy dependency)."""
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    next_label = 0
    stack: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] != target or labels[y, x] != 0:
                continue
            next_label += 1
            stack.append((y, x))
            while stack:
                cy, cx = stack.pop()
                if cy < 0 or cy >= h or cx < 0 or cx >= w:
                    continue
                if mask[cy, cx] != target or labels[cy, cx] != 0:
                    continue
                labels[cy, cx] = next_label
                stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
    return labels, next_label


def measure_cells(mask: np.ndarray, min_area: int = 50) -> list[CellMeasurement]:
    """Walk every connected cell (cytoplasm union nucleus) and report N/(N+C)."""
    fg = (mask != BACKGROUND).astype(np.uint8)
    labels, n = _label_connected(fg, target=1)
    out: list[CellMeasurement] = []
    for cell_id in range(1, n + 1):
        region = labels == cell_id
        nuc = int(((mask == NUCLEUS) & region).sum())
        cyt = int(((mask == CYTOPLASM) & region).sum())
        area = nuc + cyt
        if area < min_area:
            continue
        ratio = nuc / area if area > 0 else 0.0
        out.append(CellMeasurement(cell_id=cell_id, nucleus_area=nuc, cytoplasm_area=cyt, nc_ratio=ratio))
    return out


def field_of_view_mean(mask: np.ndarray, min_area: int = 50) -> float:
    cells = measure_cells(mask, min_area=min_area)
    if not cells:
        return 0.0
    return float(np.mean([c.nc_ratio for c in cells]))
