# Demo Video

A short walkthrough video (about 3 minutes) goes with the project so reviewers
can see the pipeline run without needing a CUDA box of their own.

## What it covers

1. Setup: cloning the repo, installing requirements, downloading the weights
   from the GitHub Release.
2. Single-image inference: running
   `python -m src.infer --image data/demo/sample.png` and looking at the
   colour overlay and the per-cell N/C ratios that get printed.
3. Evaluation: running `python -m src.evaluate` against a small slice of the
   test set and reading the JSON metrics report.
4. A quick look at `results/figures/attention_maps.png` to talk about what
   the model is actually attending to.

## Where it lives

The MP4 is uploaded alongside the model weights on the GitHub Releases page:

```
https://github.com/pg2397/bladder-cancer-vit/releases/latest
```

Once the release is published, the link goes at the top of `README.md` too.

## Re-recording

If you ever want to re-record the walkthrough, the title slides are in
`590_final_presentation_Pratik_Gajjar.pptx` in the project's parent directory.
Any screen recorder (OBS, Game Bar, QuickTime) is fine.
