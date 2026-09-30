# Open items

## Known issues

- **The fixed recipe (finding 7) goes unstable more often than not.** Of five seeds tried at `--lr 0.003 --warmup-epochs 5 --w-cov 4095`, three hit a smaller version of the epoch-25/33 instability, at severities ranging from `avg_std` 3.05 to 210.5 and at different epochs (14, 14, 17, so not schedule-tied). One never recovered (34.9% kNN); two partially recovered but plateaued well below the two clean seeds (38.5% and 40.0% kNN against 56.5% and 59.6%). See finding 13 in [EXPERIMENTS.md](EXPERIMENTS.md). The underlying cause (findings 1, 4, 10) is still not understood, only described.
- **`--adaptive` collapsed variance; the specific trap is fixed, not yet re-verified.** Magnitude balancing used to push the variance weight down whenever variance was already the largest raw loss term, which is exactly the state of a collapsed embedding, with no way back. 35.6% kNN top-1 against baseline's 56.5% at the same recipe and seed. See finding 11. **Fixed**: the variance multiplier now clips to `[1.0, 5.0]` instead of `[0.2, 5.0]`, so it can only help, never hurt (see the "Fix for finding 11" section in [EXPERIMENTS.md](EXPERIMENTS.md)). Verified against the synthetic collapsed-regime test finding 11 used; a second `--adaptive` seed with the fix hasn't been run yet, and the underlying raw-loss-value measurement (not just this one path into the trap) is still unfixed.
- **`resume_pretrain.py` didn't restore the optimizer state; fixed, not yet re-verified.** It called `safe_load_trainer_weights` before `trainer.compile()`, so the freshly built AdamW optimizer never received the checkpoint's momentum and variance state. **Fixed**: the optimizer is now compiled and its slot variables built before loading (see the "Fix for finding 12" section in EXPERIMENTS.md), confirmed with a checkpoint round-trip test. Finding 12's specific resumed run predates the fix and hasn't been re-run to see if it changes the final result.
- **`--w-cov` defaults to a value that's about D times weaker than the paper's.** `--w-cov 4095` (for `--proj-out 4096`) matches the paper's scale; the default of 1 stays for backward compatibility. See finding 6.
- **Checkpointing.** The best-loss checkpoint is chosen by training loss, not a held-out metric. The last epoch is saved too, so both ends of a run can be compared. Note it isn't crash-proof either: an interruption mid-write left one run's best-loss file corrupted, while the last-epoch file (a separate write) survived. See finding 12.
- **Full checkpoints.** The eval scripts need the encoder-only weights file. On Keras 3 a full trainer checkpoint won't load into an encoder, and `by_name` isn't supported for `.weights.h5` files. `scripts/_extract_best_encoder.py` writes an encoder-only file from a run directory.
- **Two screening runs hit NaN losses** (`--warmup-epochs 5` alone; `--clipnorm 1.0` alone), cause not isolated. `tf.math.reduce_std`'s gradient is confirmed non-finite at exact collapse (finding 14), which fits the NaN runs but not the large-but-finite spikes in findings 1, 4, 12 and 13. A `--var-eps` flag now exists (`sqrt(var + eps)`, as in reference VICReg) but hasn't been tried against a run that actually spikes.

## Experiments

- Run a second `--adaptive` seed with the finding-11 fix applied, to see whether it actually prevents the collapse in a real run and not just in the synthetic test.
- Try `--var-eps 1e-4` against a recipe known to spike (seed 1, 3 or 4's), to test whether it changes the instability rather than just fixing the two NaN cases.
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
