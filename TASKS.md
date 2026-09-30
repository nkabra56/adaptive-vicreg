# Open items

## Known issues

- **The fixed recipe (finding 7) goes unstable more often than not.** Of five seeds tried at `--lr 0.003 --warmup-epochs 5 --w-cov 4095`, three hit a smaller version of the epoch-25/33 instability, at severities ranging from `avg_std` 3.05 to 210.5 and at different epochs (14, 14, 17, so not schedule-tied). One never recovered (34.9% kNN); two partially recovered but plateaued well below the two clean seeds (38.5% and 40.0% kNN against 56.5% and 59.6%). See finding 13 in [EXPERIMENTS.md](EXPERIMENTS.md). The underlying cause (findings 1, 4, 10) is still not understood, only described.
- **`--adaptive` collapses variance.** Its magnitude balancing pushes the variance weight down and the invariance/covariance weights up to their 5x ceiling whenever variance is already the largest raw loss term, which is exactly the state of a collapsed embedding. Once collapsed there is no way back: 35.6% kNN top-1 against baseline's 56.5% at the same recipe and seed. This weight-assignment behavior is confirmed independent of the covariance weight's scale with a code-level test, though only one `--adaptive` seed has been run. See finding 11. A fix candidate: clip the variance multiplier to `[1.0, 5.0]` instead of `[0.2, 5.0]` so it can only help, never hurt, though that only closes this one path and leaves the underlying raw-loss-value measurement unfixed.
- **`resume_pretrain.py` doesn't actually restore the optimizer state.** It calls `safe_load_trainer_weights` before `trainer.compile()`, so the freshly built AdamW optimizer never receives the checkpoint's momentum and variance state; a "resumed" run's optimizer restarts cold. The loss curve stayed smooth across one resume, but that can't rule out a subtler effect on the final result, and there's no non-resumed control to check it against. See finding 12. Fix: compile before loading, or load into an already-compiled trainer.
- **`--w-cov` defaults to a value that's about D times weaker than the paper's.** `--w-cov 4095` (for `--proj-out 4096`) matches the paper's scale; the default of 1 stays for backward compatibility. See finding 6.
- **Checkpointing.** The best-loss checkpoint is chosen by training loss, not a held-out metric. The last epoch is saved too, so both ends of a run can be compared. Note it isn't crash-proof either: an interruption mid-write left one run's best-loss file corrupted, while the last-epoch file (a separate write) survived. See finding 12.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.
- **Two screening runs hit NaN losses** (`--warmup-epochs 5` alone; `--clipnorm 1.0` alone), cause not isolated. One untested hypothesis: `variance_loss` uses `tf.math.reduce_std` with no epsilon, unlike reference VICReg's `sqrt(var + 1e-4)`. See finding 10.

## Experiments

- Run a second `--adaptive` seed, since only one has been run and finding 13 shows the baseline recipe's collapse behavior is seed-sensitive.
- Test batch size on its own as a variable in the instability (finding 13): every run so far used 256, an 8 GB GPU limit, against the paper's much larger batches.
- Ablate `--adaptive-targets`. Its default `nu` schedule starts at 1.0, which pulls off-diagonal correlation toward redundancy early in training.
- Feed the final comparison runs into `report_metrics.py` for the plots and tables described in the README.

## Scope and tooling

- Only CIFAR-10 and CIFAR-100 are supported (the 50,000-image count is hardcoded in `steps_for_dataset`).
- Augmentation is minimal: no random-resized-crop, Gaussian blur or solarization.
- There is one fixed CNN encoder, with no ResNet option and no LARS optimizer (the VICReg paper uses LARS for large batches).
- No CI, so the tests only run when someone runs them.
- `report_metrics.py` is a command-line script that lives inside the `vicreg_tf` package. `scripts/` would suit it better.
- `VicRegMetricsLogger` and `AdaptiveReweighter` each compute the embedding std and off-diagonal correlation on their own. They could share code, but merging them can change float rounding, so check that seeded runs still match first.
- `citation.cff` has no ORCID. Add one if it should be in the citation record.
