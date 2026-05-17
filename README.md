# Automated Bladder Cancer Screening using Vision Transformer (ViT)

Vision Transformer + U-Net pipeline for segmenting urine cytology images into
background, cytoplasm, and nucleus, then computing per-cell nucleus-to-cytoplasm
(N/C) ratios. Built as my CSUDH M.S. Computer Science Master's Project, with
industry mentorship from Cedars-Sinai Medical Center.

## Results on the held-out test split

| Method                 | Pixel Acc | Macro F1 | Nucleus F1 | Spearman r | MAE   |
|------------------------|----------:|---------:|-----------:|-----------:|------:|
| Multi-level Otsu       |     0.79  |    0.51  |      0.32  |     0.707  | 0.094 |
| K-means (RGB)          |     0.83  |    0.62  |      0.48  |     0.768  | 0.074 |
| ViT + U-Net (ours)     |     0.92  |    0.84  |      0.76  |     0.959  | 0.025 |

On the clinically relevant ranking task, the ViT model scores Spearman r = 0.855
against the Paris System label, essentially tied with the cytopathologist
reference (0.851). The photometric-augmentation ablation is a negative result
that I report honestly in Section 5.6 of the report.

## Layout

```
.
├── src/                Python modules (data, model, training, eval, inference, baselines)
├── notebooks/          Jupyter notebooks from the experiments
├── configs/            YAML configs for the four runs in Chapter 5
├── results/
│   ├── figures/        PNGs used in the report
│   ├── logs/           Training log from the augmentation run
│   └── metrics/        Per-run metric JSONs
├── data/demo/          Small synthetic image so the inference path runs OOTB
├── docs/               Data access policy, weights instructions, demo video link
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
git clone https://github.com/pg2397/bladder-cancer-vit.git
cd bladder-cancer-vit
python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.1 on an RTX 3080 (10 GB). CPU inference
works but is roughly 30x slower.

### Download the trained weights

The two checkpoints are 350 MB each, so they live on the GitHub Releases page
rather than in the tree.

```bash
mkdir -p results/weights
curl -L -o results/weights/vit_seg_best.pth \
  https://github.com/pg2397/bladder-cancer-vit/releases/latest/download/vit_seg_best.pth
curl -L -o results/weights/vit_seg_augmented_best.pth \
  https://github.com/pg2397/bladder-cancer-vit/releases/latest/download/vit_seg_augmented_best.pth
```

See `docs/WEIGHTS.md` for details on what each file is.

### Single-image inference

```bash
python -m src.infer \
  --config configs/vit.yaml \
  --weights results/weights/vit_seg_best.pth \
  --image data/demo/sample.png \
  --out results/demo_output.png
```

This segments the image into three classes, writes a colour-coded overlay, and
prints the per-cell N/C ratios.

### Reproduce the reported metrics

```bash
python -m src.evaluate \
  --config configs/vit.yaml \
  --weights results/weights/vit_seg_best.pth \
  --report-json results/metrics/vit_test.json

python -m src.baselines.otsu   --config configs/otsu.yaml   --report-json results/metrics/otsu_test.json
python -m src.baselines.kmeans --config configs/kmeans.yaml --report-json results/metrics/kmeans_test.json
```

### Train from scratch

```bash
python -m src.train --config configs/vit.yaml             # main model, ~3 h on RTX 3080
python -m src.train --config configs/vit_augmented.yaml   # ablation with photometric augmentation
```

## Reproducibility

Each experiment is fully described by its YAML config. Every source of
randomness (Python `random`, NumPy, PyTorch CPU, CUDA, and the per-worker
DataLoader generators) is seeded from the config. With the same config, same
seed, and same hardware, training reproduces the reported numbers to within
numerical tolerance. The hyperparameter table in Appendix C of the report
matches `configs/vit.yaml` line for line.

## Dataset

The training and evaluation images are de-identified urine cytology slides
provided under a Cedars-Sinai data-use agreement. The DUA forbids public
redistribution, so the full dataset is not part of this repo. What is shipped:

- `data/demo/sample.png`, a small synthetic image that exercises the inference
  path without needing any patient data.

Researchers who need the real dataset for replication should follow the process
in `docs/DATA_ACCESS.md`.

## Citation

```bibtex
@mastersthesis{gajjar2026vit_bladder,
  author  = {Pratik Nileshkumar Gajjar},
  title   = {Automated Bladder Cancer Screening using Vision Transformer (ViT)},
  school  = {California State University, Dominguez Hills},
  year    = {2026},
  type    = {Master's Project Report},
  note    = {Industry partner: Cedars-Sinai Medical Center}
}
```

`CITATION.cff` has the machine-readable version.

## License

Source code is MIT, see `LICENSE`. The trained weights are released for research
and educational use only and must not be used clinically without proper
regulatory clearance. The full notice is at the bottom of `LICENSE` and in
`docs/WEIGHTS.md`.

## Acknowledgements

Thanks to my advisors at Cedars-Sinai for the mentorship, the slide material,
and the cytopathologist reference annotations. The ViT-Base/16 backbone comes
from the [timm](https://github.com/rwightman/pytorch-image-models) library.
