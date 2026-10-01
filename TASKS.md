# Open items

## Known issues

- **The fixed recipe (finding 7) goes unstable more often than not.** Of five seeds tried at `--lr 0.003 --warmup-epochs 5 --w-cov 4095`, three hit a smaller version of the epoch-25/33 instability, at severities ranging from `avg_std` 3.05 to 210.5 and at different epochs (14, 14, 17, so not schedule-tied). One never recovered (34.9% kNN); two partially recovered but plateaued well below the two clean seeds (38.5% and 40.0% kNN against 56.5% and 59.6%). See finding 13 in [EXPERIMENTS.md](EXPERIMENTS.md). The underlying cause (findings 1, 4, 10) is still not understood, only described.
- **`--adaptive` no longer collapses, and the reweighter redesign is now validated on real training.** Trend-relative balancing (`--trend-decay`, finding 20) compares each term only to its own slower-moving EMA instead of to the other two's current values, so a term is only boosted when it's actually trending worse, never punished for being large by comparison. Two full 100-epoch runs (finding 21): seed 1 lands within about 2 points of clean baseline (60% linear / 53% kNN against baseline's 61-62% / 56-60%), seed 0 improves substantially (53.6%/47.4%) but not all the way. The gap isn't fully closed in every seed; `--trend-decay`'s default (0.995) hasn't been tuned.
- **`--adaptive-targets`'s apparent underperformance was mostly a checkpoint-selection artifact.** Its decaying `nu` makes `l_cov`, and so total loss, rise as decorrelation genuinely improves, so `monitor="loss"` picks a nearly redundant early epoch (epoch 6, `corr^2` 0.989, in one seed) over the well-decorrelated final one (epoch 100, `corr^2` 0.0055). Read from the last epoch instead of best-loss, both seeds land close to baseline (57-59% linear / 52-53% kNN). See finding 16. This is a sharper version of the general checkpointing issue below.
- **`resume_pretrain.py`'s optimizer-reset fix is confirmed at the mechanism level (checkpoint round-trip test), not yet re-run against the original resumed seed.**
- **The instability's root cause is localized but still unknown.** Per-epoch data (finding 19) ruled out any fixed threshold, step-count clustering, and a weighted-covariance anomaly as leading indicators. A `--per-batch-diagnostics` run then caught a real event directly (finding 22): a whole-embedding explosion (`avg_std` and `min_std` both jump together, so not a single collapsing dimension) building over about 30 batches, not a sudden one-batch jump. The variance+covariance gradient on the probe batch, the one signal measured, actually *falls* during the blowup, so it isn't what's driving it. Two real gaps in the diagnostic (no invariance-loss gradient, probe batch instead of the real training batch) are the natural next thing to close. Also noticed: `train_vicreg.py` never uses the `LossExplosionGuard` callback that `resume_pretrain.py` does; every documented spike ran with no such guard active.
- **Batch size 128 doesn't prevent the instability either.** One of two seeds failed. Finding 19 corrects finding 17: this failure actually included a sharp spike (epoch 11, `loss/total` up 5.9x) overlapping with a separate gradual `avg_std` decline, not a purely gradual failure as originally described.
- **`--w-cov` defaults to a value that's about D times weaker than the paper's.** `--w-cov 4095` (for `--proj-out 4096`) matches the paper's scale; the default of 1 stays for backward compatibility. See finding 6.
- **Checkpointing.** The best-loss checkpoint is chosen by training loss, not a held-out metric, and finding 16 shows this can be actively misleading (not just uninformative) when a run's loss target changes over time. The last epoch is saved too, so both ends of a run can be compared. Note it isn't crash-proof either: an interruption mid-write left one run's best-loss file corrupted, while the last-epoch file (a separate write) survived. See finding 12.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.

## Experiments

- Close the two gaps in `--per-batch-diagnostics` (finding 22): measure the invariance loss's gradient too, and the real training batch's gradient, not just the fixed probe's. Then target the same ~30-batch window with `--per-batch-diagnostics-every 1` and `--stop-epoch` around the known event.
- Tune or seed-sweep `--trend-decay`. Finding 21's two seeds landed at different distances from baseline; unclear if that's seed noise or the untuned 0.995 default.
- **Done: fed all 12 tracked runs into `report_metrics.py`**, output at `reports/full_comparison/` (gitignored, regenerate with the command in EXPERIMENTS.md's "Comparison report" section).

## Scope and tooling

- Only CIFAR-10 and CIFAR-100 are supported (the 50,000-image count is hardcoded in `steps_for_dataset`).
- Augmentation is minimal: no random-resized-crop, Gaussian blur or solarization.
- There is one fixed CNN encoder, with no ResNet option and no LARS optimizer (the VICReg paper uses LARS for large batches).
- No CI, so the tests only run when someone runs them.
- `report_metrics.py` is a command-line script that lives inside the `vicreg_tf` package. `scripts/` would suit it better.
- `VicRegMetricsLogger` and `AdaptiveReweighter` each compute the embedding std and off-diagonal correlation on their own. They could share code, but merging them can change float rounding, so check that seeded runs still match first.
- `citation.cff` has no ORCID. Add one if it should be in the citation record.
