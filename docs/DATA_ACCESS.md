# Data Access

The training data is a set of de-identified urine cytology images prepared by
the host laboratory at Cedars-Sinai Medical Center. It is not public and is
not included in this repository.

## Why it isn't here

- The slides come from clinical patient material.
- Even de-identified, redistribution would need a Data Use Agreement (DUA)
  between the requesting institution and Cedars-Sinai, which is outside the
  scope of a student project.
- The HIPAA Privacy Rule and the Cedars-Sinai IRB protocol covering this work
  restrict secondary use to approved research personnel.

## What is in the repo

- A small synthetic demo image at `data/demo/sample.png` so the inference path
  runs end to end without any patient data.
- All evaluation metrics in `results/metrics/`.
- The trained model weights, via GitHub Releases (see `docs/WEIGHTS.md`).

This is enough to replicate the inference and evaluation halves of the work
without anyone needing to transfer patient-derived images.

## Requesting access for replication

Researchers who want to reproduce the training results should contact the
Cedars-Sinai Department of Pathology and Laboratory Medicine through their
own institution's IRB office. A new DUA needs to be in place before any data
transfer can happen.

## Expected directory layout

The dataset, once obtained, should be on disk in the layout `CytologyDataset`
expects:

```
<data_root>/
├── images/
│   ├── case_0001.png
│   ├── case_0002.png
│   └── ...
├── masks/
│   ├── case_0001.png      # uint8, values in {0, 1, 2}
│   └── ...
└── splits.json            # {"train": [...], "val": [...], "test": [...]}
```

If `splits.json` is missing, the dataset falls back to a deterministic 80/10/10
split keyed by the experiment seed in the config.

## Per-image specs

- Field of view: 224 x 224 pixels (resized from the 1024 x 1024 native crop).
- Stain: Papanicolaou.
- Mask classes: 0 = background, 1 = cytoplasm, 2 = nucleus.
- Annotators: two trained graduate students with cytopathologist review
  (Section 3.9 of the report).
