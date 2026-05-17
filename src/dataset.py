"""Urine cytology dataset wrapper.

Loads cytology image / mask pairs and applies the geometric and (optionally)
photometric augmentations described in Sections 3.6 and 4.10 of the report.
The dataset is expected to be a pickle on disk produced by the Cedars-Sinai
preprocessing pipeline; see ``data/cell_specimens_data/`` and the project
``docs/DATA_ACCESS.md`` note.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


# Three semantic classes in the masks.
BACKGROUND, CYTOPLASM, NUCLEUS = 0, 1, 2
NUM_CLASSES = 3


class CytologyDataset(Dataset):
    """Triplets of (image, mask, slide_id) loaded from a single pickle."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        cfg: Mapping[str, Any] | None = None,
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.cfg = cfg or {}
        self.transform = transform
        self._items = self._load_split()

    # ----------------------------------------------------------- loading
    def _load_split(self) -> list[dict[str, Any]]:
        pkl_candidates = sorted(self.root.glob("*.pkl"))
        if not pkl_candidates:
            raise FileNotFoundError(
                f"No .pkl files found under {self.root}. "
                "See docs/DATA_ACCESS.md for how to obtain the full dataset."
            )
        # Prefer the explicit per-cell file when present.
        primary = [p for p in pkl_candidates if "urothelial" in p.name.lower()]
        pkl_path = primary[0] if primary else pkl_candidates[0]
        with pkl_path.open("rb") as f:
            blob = pickle.load(f)

        items = blob.get(self.split) if isinstance(blob, Mapping) else None
        if items is None:
            # Fall back to a deterministic split derived from the seed below.
            items = self._derive_split(blob)
        return list(items)

    def _derive_split(self, blob: Any) -> list[dict[str, Any]]:
        """Reproducibly slice a single list into train / val / test."""
        from .seed import seed_everything
        seed = int(self.cfg.get("experiment", {}).get("seed", 42))
        seed_everything(seed)

        if isinstance(blob, Mapping) and "items" in blob:
            items = list(blob["items"])
        elif isinstance(blob, list):
            items = list(blob)
        else:
            raise TypeError(f"Unsupported dataset blob type: {type(blob)!r}")

        # Deterministic shuffle.
        rng = np.random.default_rng(seed)
        idx = np.arange(len(items))
        rng.shuffle(idx)

        data_cfg = self.cfg.get("data", {})
        n = len(items)
        n_tr = int(n * float(data_cfg.get("train_split", 0.70)))
        n_va = int(n * float(data_cfg.get("val_split", 0.15)))
        if self.split == "train":
            sel = idx[:n_tr]
        elif self.split == "val":
            sel = idx[n_tr:n_tr + n_va]
        else:
            sel = idx[n_tr + n_va:]
        return [items[i] for i in sel]

    # ----------------------------------------------------------- API
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self._items[i]
        image = np.asarray(item["image"], dtype=np.float32)  # H x W x 3, 0..255
        mask = np.asarray(item["mask"], dtype=np.int64)       # H x W in {0,1,2}

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # ImageNet normalisation, see configs/vit.yaml.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        image = (image - mean) / std

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()
        return {"image": image_t, "mask": mask_t, "slide_id": item.get("slide_id", "")}


def compute_class_weights(loader, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to sum to ``num_classes``.

    Computed per-batch and accumulated across the loader to avoid the
    out-of-memory failure described in Section 4.12 of the report.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in loader:
        m = batch["mask"]
        for c in range(num_classes):
            counts[c] += (m == c).sum().item()
    inv = 1.0 / (counts + 1e-9)
    weights = inv / inv.sum() * num_classes
    return weights.float()
