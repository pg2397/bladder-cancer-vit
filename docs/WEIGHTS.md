# Pretrained Weights

The two ViT-Base/16 segmentation checkpoints from Chapter 5 are about 350 MB
each, so I put them on the GitHub Releases page instead of in the tree.

## Files

| File                         | Size     | Macro F1 (test) | Source config              |
|------------------------------|---------:|----------------:|----------------------------|
| `vit_seg_best.pth`           | ~350 MB  |           0.86  | `configs/vit.yaml`         |
| `vit_seg_augmented_best.pth` | ~350 MB  |           0.89  | `configs/vit_augmented.yaml` |

Both are weighted averages of the top-3 checkpoints by validation macro-F1, as
described in Section 4.12 of the report.

## Download

Once the first release is published:

```bash
mkdir -p results/weights
curl -L -o results/weights/vit_seg_best.pth \
  https://github.com/pg2397/bladder-cancer-vit/releases/latest/download/vit_seg_best.pth
curl -L -o results/weights/vit_seg_augmented_best.pth \
  https://github.com/pg2397/bladder-cancer-vit/releases/latest/download/vit_seg_augmented_best.pth
```

## Format

Each `.pth` file is a dict with one key:

```python
state = torch.load("vit_seg_best.pth", map_location="cpu")
state.keys()   # dict_keys(['model'])
```

`state["model"]` is a standard PyTorch state dict that loads into
`src.model.ViTUNet` with `strict=True`. The evaluator and the inference script
both unwrap this shape via `state = state.get("model", state)`, so a raw state
dict also works if you produced one elsewhere.

## Reproducing the weights

```bash
python -m src.train --config configs/vit.yaml             # baseline
python -m src.train --config configs/vit_augmented.yaml   # augmented
```

Around 6 hours per config on a single A100 (40 GB), or roughly 3 hours on an
RTX 3080 if you drop the batch size.

## Use restrictions

Trained on a small academic dataset under a clinical DUA. Released for research
and educational use only. Do not use them to make clinical decisions without
proper validation and regulatory clearance.
