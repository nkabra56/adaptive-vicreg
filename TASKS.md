# Open items

## Known issues

- **The fixed recipe (finding 7) goes unstable more often than not.** Of five seeds tried at `--lr 0.003 --warmup-epochs 5 --w-cov 4095`, three hit a smaller version of the epoch-25/33 instability, at severities ranging from `avg_std` 3.05 to 210.5 and at different epochs (14, 14, 17, so not schedule-tied). One never recovered (34.9% kNN); two partially recovered but plateaued well below the two clean seeds (38.5% and 40.0% kNN against 56.5% and 59.6%). See finding 13 in [EXPERIMENTS.md](EXPERIMENTS.md). The underlying cause (findings 1, 4, 10) is still not understood, only described.
- **`--adaptive` no longer collapses, but still trails baseline.** The variance-multiplier floor fix (`[1.0, 5.0]` instead of `[0.2, 5.0]`) is confirmed in two full runs: neither seed collapsed (finding 15). But magnitude balancing now suppresses the invariance weight instead, to about 0.34x, once that becomes the largest raw term, so both seeds land at 42-46% linear / 39-45% kNN, well below baseline's 56-62%. The reweighter's core design (raw loss value decides weight, regardless of whether that term still needs optimizing) is the real issue, not just the variance floor. See finding 15.
- **`--adaptive-targets`'s apparent underperformance was mostly a checkpoint-selection artifact.** Its decaying `nu` makes `l_cov`, and so total loss, rise as decorrelation genuinely improves, so `monitor="loss"` picks a nearly redundant early epoch (epoch 6, `corr^2` 0.989, in one seed) over the well-decorrelated final one (epoch 100, `corr^2` 0.0055). Read from the last epoch instead of best-loss, both seeds land close to baseline (57-59% linear / 52-53% kNN). See finding 16. This is a sharper version of the general checkpointing issue below.
- **`resume_pretrain.py`'s optimizer-reset fix is confirmed at the mechanism level (checkpoint round-trip test), not yet re-run against the original resumed seed.**
- **The instability's root cause is still open after ruling out one hypothesis.** `--var-eps 1e-4`, re-run against seed 3's and seed 4's exact recipes, did not prevent either spike (finding 18): both hit the same epoch-14 event, one worse and one better than without the epsilon. The epsilon only fixes the non-finite gradient at *exact* collapse (finding 10's two NaN runs), not whatever causes the large-but-finite spikes in findings 1, 4, 12, 13, 17 and 18.
- **Batch size 128 doesn't prevent the instability either, and can fail differently.** One of two seeds failed, but gradually (falling `avg_std`, rising `l_var` over many epochs) rather than in one sharp spike (finding 17). Two seeds is too few to compare failure rates against batch 256's.
- **`--w-cov` defaults to a value that's about D times weaker than the paper's.** `--w-cov 4095` (for `--proj-out 4096`) matches the paper's scale; the default of 1 stays for backward compatibility. See finding 6.
- **Checkpointing.** The best-loss checkpoint is chosen by training loss, not a held-out metric, and finding 16 shows this can be actively misleading (not just uninformative) when a run's loss target changes over time. The last epoch is saved too, so both ends of a run can be compared. Note it isn't crash-proof either: an interruption mid-write left one run's best-loss file corrupted, while the last-epoch file (a separate write) survived. See finding 12.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.

## Experiments

- Redesign `AdaptiveReweighter`'s magnitude balancing so it reflects whether a term still needs optimizing, not just its current raw loss value. Closing the variance floor (finding 15) just moved the same failure onto the invariance term.
- Find the instability's actual root cause. Ruled out so far: the `--var-eps` epsilon hypothesis (finding 18) and (on 2 seeds) batch size 128 (finding 17). Nothing tried yet has stopped it, only changed its shape or severity.
- **Done: fed all 12 tracked runs into `report_metrics.py`**, output at `reports/full_comparison/` (gitignored, regenerate with the command in EXPERIMENTS.md's "Comparison report" section).

## Scope and tooling

- Only CIFAR-10 and CIFAR-100 are supported (the 50,000-image count is hardcoded in `steps_for_dataset`).
- Augmentation is minimal: no random-resized-crop, Gaussian blur or solarization.
- There is one fixed CNN encoder, with no ResNet option and no LARS optimizer (the VICReg paper uses LARS for large batches).
- No CI, so the tests only run when someone runs them.
- `report_metrics.py` is a command-line script that lives inside the `vicreg_tf` package. `scripts/` would suit it better.
- `VicRegMetricsLogger` and `AdaptiveReweighter` each compute the embedding std and off-diagonal correlation on their own. They could share code, but merging them can change float rounding, so check that seeded runs still match first.
- `citation.cff` has no ORCID. Add one if it should be in the citation record.
