# Open items

## Known issues

- **Resuming.** `resume_pretrain.py --use-schedules` has no effect: no LR schedule is attached, so the rate stays constant after warmup. `--warmup-steps` is compared against the global step, which a resume sets to `initial_epoch * steps_per_epoch`, so warmup only happens when it is larger than that.
- **Unstable baseline.** At batch size 256 the baseline recipe blows up around epoch 25 to 33 (see [EXPERIMENTS.md](EXPERIMENTS.md)). Things to try: a lower peak LR, a real warmup (`CosineWarmup` is built with `warmup_frac=0.0`), gradient clipping.
- **Checkpointing.** `ModelCheckpoint` monitors the training loss, not a held-out metric, and keeps only the best-loss weights. Add an unconditional last-epoch checkpoint so both ends of a run can be compared.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.

## Experiments

- Benchmark `--adaptive` against a stable baseline. The numbers in the README predate the current reweighter.
- Ablate `--adaptive-targets`. Its default `nu` schedule starts at 1.0, which pulls off-diagonal correlation toward redundancy early in training.

## Scope and tooling

- Only CIFAR-10 and CIFAR-100 are supported (the 50,000-image count is hardcoded in `steps_for_dataset`).
- Augmentation is minimal: no random-resized-crop, Gaussian blur or solarization.
- There is one fixed CNN encoder, with no ResNet option and no LARS optimizer (the VICReg paper uses LARS for large batches).
- No CI, so the tests only run when someone runs them.
- `report_metrics.py` is a command-line script that lives inside the `vicreg_tf` package. `scripts/` would suit it better.
- `VicRegMetricsLogger` and `AdaptiveReweighter` each compute the embedding std and off-diagonal correlation on their own. They could share code, but merging them can change float rounding, so check that seeded runs still match first.
- `citation.cff` has no ORCID. Add one if it should be in the citation record.
