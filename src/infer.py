"""Single-image inference + N/C-ratio computation.

Usage:
    python -m src.infer --config configs/vit.yaml \
                        --weights results/weights/vit_seg_best.pth \
                        --image data/demo/sample.png \
                        --out results/demo_output.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .dataset import NUM_CLASSES
from .model import build_model
from .nc_ratio import measure_cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="results/demo_output.png")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], num_classes=NUM_CLASSES).to(device)
    state = torch.load(args.weights, map_location=device)
    state = state.get("model", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load + ImageNet-normalise.
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required: pip install Pillow") from e
    img_pil = Image.open(args.image).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img_pil, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    cells = measure_cells(pred)
    print(f"[infer] cells found: {len(cells)}")
    for c in cells[:20]:
        print(f"  cell {c.cell_id:>3d}  nuc={c.nucleus_area:>5d}  "
              f"cyt={c.cytoplasm_area:>5d}  N/C={c.nc_ratio:.3f}")

    # Save a colour-coded overlay.
    _save_overlay(arr, pred, Path(args.out))


def _save_overlay(image: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Background -> dim, cytoplasm -> blue tint, nucleus -> red tint."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = image.astype(np.float32)
    overlay = img.copy()
    overlay[mask == 1] = 0.5 * img[mask == 1] + 0.5 * np.array([60.0, 100.0, 220.0])
    overlay[mask == 2] = 0.4 * img[mask == 2] + 0.6 * np.array([220.0, 60.0, 60.0])
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(overlay).save(out_path)
        print(f"[infer] overlay saved -> {out_path}")
    except ImportError:
        np.save(out_path.with_suffix(".npy"), overlay)


if __name__ == "__main__":
    main()
