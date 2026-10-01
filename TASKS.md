# Open items

## Known issues

- **The fixed recipe (finding 7) goes unstable more often than not.** Of five seeds tried at `--lr 0.003 --warmup-epochs 5 --w-cov 4095`, three hit a smaller version of the epoch-25/33 instability, at severities ranging from `avg_std` 3.05 to 210.5 and at different epochs (14, 14, 17, so not schedule-tied). One never recovered (34.9% kNN); two partially recovered but plateaued well below the two clean seeds (38.5% and 40.0% kNN against 56.5% and 59.6%). See finding 13 in [EXPERIMENTS.md](EXPERIMENTS.md). The underlying cause (findings 1, 4, 10) is still not understood, only described.
- **`--adaptive` no longer collapses, but still trailed baseline; the reweighter has been redesigned, not yet re-run.** The old magnitude balancing compared each term against the other two's current EMAs, so whichever was currently largest always got punished, which is what caused both finding 11 (variance, when collapsed) and finding 15 (invariance, once variance/covariance converged and became the new "smallest," making align relatively the largest). **Fixed**: balancing now compares each term only to its own slower-moving EMA (`trend_decay`, new flag), so a term is only boosted when it's actually trending worse, never punished just for being large by comparison. Verified in synthetic tests reproducing both findings' exact regimes (finding 20 in [EXPERIMENTS.md](EXPERIMENTS.md)); no real GPU run has used it yet.
- **`--adaptive-targets`'s apparent underperformance was mostly a checkpoint-selection artifact.** Its decaying `nu` makes `l_cov`, and so total loss, rise as decorrelation genuinely improves, so `monitor="loss"` picks a nearly redundant early epoch (epoch 6, `corr^2` 0.989, in one seed) over the well-decorrelated final one (epoch 100, `corr^2` 0.0055). Read from the last epoch instead of best-loss, both seeds land close to baseline (57-59% linear / 52-53% kNN). See finding 16. This is a sharper version of the general checkpointing issue below.
- **`resume_pretrain.py`'s optimizer-reset fix is confirmed at the mechanism level (checkpoint round-trip test), not yet re-run against the original resumed seed.**
- **The instability's root cause is still open, but per-epoch hypotheses are exhausted.** A systematic pass over all 13 existing run histories (finding 19) ruled out any fixed per-epoch threshold, step-count clustering across batch sizes, and a weighted-covariance anomaly as leading indicators; the only signal found (an accelerating loss/align trend) is present in clean runs too, just smaller, and gives under one epoch of lead time. The trigger almost certainly fires within a single epoch's batches. A `PerBatchDiagnosticsLogger` (`--per-batch-diagnostics` in `train_vicreg.py`) is built and ready to catch it; no GPU run has used it yet.
- **Batch size 128 doesn't prevent the instability either.** One of two seeds failed. Finding 19 corrects finding 17: this failure actually included a sharp spike (epoch 11, `loss/total` up 5.9x) overlapping with a separate gradual `avg_std` decline, not a purely gradual failure as originally described.
- **`--w-cov` defaults to a value that's about D times weaker than the paper's.** `--w-cov 4095` (for `--proj-out 4096`) matches the paper's scale; the default of 1 stays for backward compatibility. See finding 6.
- **Checkpointing.** The best-loss checkpoint is chosen by training loss, not a held-out metric, and finding 16 shows this can be actively misleading (not just uninformative) when a run's loss target changes over time. The last epoch is saved too, so both ends of a run can be compared. Note it isn't crash-proof either: an interruption mid-write left one run's best-loss file corrupted, while the last-epoch file (a separate write) survived. See finding 12.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.

## Experiments

- Run a full `--adaptive` job (or two, matching findings 11/15's seeds) with the redesigned reweighter, to see whether finding 20's synthetic-test fix holds up on real training and actually recovers baseline-level accuracy.
- Run a job with `--per-batch-diagnostics` against seed 1's or seed 3's recipe (both spiked at epoch 14), to try to catch the trigger within the epoch it actually happens in. Command is in `EXPERIMENTS_instability_investigation.md`.
- **Done: fed all 12 tracked runs into `report_metrics.py`**, output at `reports/full_comparison/` (gitignored, regenerate with the command in EXPERIMENTS.md's "Comparison report" section).

## Scope and tooling

- Only CIFAR-10 and CIFAR-100 are supported (the 50,000-image count is hardcoded in `steps_for_dataset`).
- Augmentation is minimal: no random-resized-crop, Gaussian blur or solarization.
- There is one fixed CNN encoder, with no ResNet option and no LARS optimizer (the VICReg paper uses LARS for large batches).
- No CI, so the tests only run when someone runs them.
- `report_metrics.py` is a command-line script that lives inside the `vicreg_tf` package. `scripts/` would suit it better.
- `VicRegMetricsLogger` and `AdaptiveReweighter` each compute the embedding std and off-diagonal correlation on their own. They could share code, but merging them can change float rounding, so check that seeded runs still match first.
- `citation.cff` has no ORCID. Add one if it should be in the citation record.
