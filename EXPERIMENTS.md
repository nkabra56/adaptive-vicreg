# Experiment log

Local runs on a Windows 11 laptop in the NVIDIA NGC TensorFlow container (see `Dockerfile`; the local helper scripts `ngcContEnv.sh` and `ngcContainer.sh` are not tracked).

## Environment (2026-09-08)

- **Host:** Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, compute capability 12.0 / sm_120), driver 610.88, Docker Desktop with the WSL2 backend.
- **Container:** `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` (TensorFlow 2.17.0, CUDA 12.8.0, cuDNN 9.7.1). This is NVIDIA's last TF NGC release; the series was discontinued after this tag. The repo's `requirements.txt` is installed on top through the `Dockerfile`, with `protobuf` pinned to `<5.0.0dev,>=3.20.3`. `tensorflow-datasets` (since removed from the requirements) upgraded protobuf to 7.x, which breaks this TF build.
- **Dataset cache:** `.cache_keras/` is bind-mounted to `/root/.keras`, so CIFAR-10 is downloaded once. It comes from `keras.datasets.cifar10`, not `tensorflow-datasets`. The first download took about 20 minutes through Docker Desktop's NAT (~130 KB/s) and is cached on the host now.

## Calibration

- The README's `--batch-size 512` (with `--feat-dim 2048 --proj-out 4096 --proj-layers 3`) runs out of memory on this 8 GB GPU. The error is `RESOURCE_EXHAUSTED` during the variance-loss gradient (`reduce_variance/Tile` on a `[512, 4096]` tensor): the 4096-wide, 3-layer projector's activations and gradients don't fit next to the two-view forward pass.
- Mixed precision exists (`utils.set_mixed_precision`), but `train_vicreg.py` turns it off on purpose ("float32 for stability with VICReg"), so it stays off.
- `--batch-size 256` with the same feat-dim, proj-out and proj-layers fits: about 3.5 GB of 8 GB, 96% GPU utilization, ~137 ms/step, 195 steps/epoch, ~34 s/epoch. That is about 57 minutes for 100 epochs.
- `--batch-size 128` also worked in an earlier 2-epoch smoke test (~80-100 ms/step).

## Run 1: `pretrain-c10_baseline_20260908-1353` (baseline VICReg)

A full-length baseline run, as the reference for a later Adaptive VICReg comparison.

- **Started:** 2026-09-08 08:52 local. The container was named `pretrain-c10_baseline_20260908-0852`. The script's own timestamp, used for the checkpoint directory, is `20260908-1353` because the container clock is offset from host local time. Both names refer to the same run.
- **Command:**
  ```bash
  docker run --gpus all -d --name pretrain-c10_baseline_20260908-0852 \
    -v "$(pwd):/workspace" -v "$(pwd)/.cache_keras:/root/.keras" -w /workspace \
    adaptive-vicreg:ngc \
    python3 -u scripts/train_vicreg.py \
      --dataset cifar10 --image-size 32 --epochs 100 --batch-size 256 \
      --feat-dim 2048 --proj-out 4096 --proj-layers 3 \
      --lr 0.01 --wd 1e-6 --use-schedules \
      --metrics-compute-on projector --metrics-probe-batch 256 --record-every 1 \
      --model-dir checkpoints_tf --run-name pretrain-c10_baseline
  ```
- **Deviation from the README recipe:** `--batch-size 256` instead of `512` (see Calibration). Every other hyperparameter matches the README's baseline command.
- **Output:** `checkpoints_tf/pretrain-c10_baseline_20260908-1353/` with `vicreg_full.weights.h5`, `vicreg_encoder.weights.h5`, `vicreg_encoder_bestckpt.weights.h5` (see finding 2), `train_config.json` and `metrics/history.jsonl`.
- **Status:** all 100 epochs finished (exit code 0), but the run is not healthy (see the findings). About 57 minutes of pretraining plus a few minutes of evaluation.

### Finding 1: training becomes unstable at epoch 25

The history in `metrics/history.jsonl` is clean through epoch 24: total loss 4.5-4.6 and `avg_std` (mean per-dimension feature std) around 1.0-1.4. At epoch 25, `avg_std` jumps to 14,767 and `loss/align` to 16.7 million. The total loss never returns to its earlier level. It decays over about 50 epochs and plateaus near 35 for the rest of training, against a best of 4.49 at epoch 19.

`variance_loss` is `relu(gamma - std)`, so it is zero once `std >= gamma`. Nothing in the objective penalizes std for growing far above the floor. Once something pushes the encoder into a runaway-magnitude regime (a large gradient step, probably, since the LR is still high this early in the cosine schedule), nothing pulls it back. `TerminateOnNaN()` never fired because the values stayed finite.

Whether this depends on this batch size, LR and hardware, or is a general weakness of the baseline recipe, is open. Worth ablating: does it also happen at batch 512 on other hardware, and does it depend on where the LR schedule is?

### Finding 2: `vicreg_encoder.weights.h5` did not hold the best checkpoint (code bug)

`ModelCheckpoint(monitor="loss", save_best_only=True)` kept `vicreg_full.weights.h5` at its epoch-19 best. File mtimes and the "did not improve from 4.49050" message for every later epoch confirm it. But right after `trainer.fit()` returned, the script called `encoder.save_weights(enc_ckpt)` on whatever was in memory at the final epoch and never reloaded `vicreg_full.weights.h5`. Every eval script reads `vicreg_encoder.weights.h5`, so this silently shipped the post-collapse epoch-100 encoder instead of the best one, with no error or warning.

Fix: reload the best full checkpoint into the trainer (`safe_load_trainer_weights`, already in `utils.py`) before the final `encoder.save_weights(...)`.

To measure the impact and get a second data point, `scripts/_extract_best_encoder.py` reloads `vicreg_full.weights.h5` into a fresh trainer and re-saves the encoder as `vicreg_encoder_bestckpt.weights.h5`.

### Finding 3: even the best checkpoint (epoch 19) is undertrained and still redundant

At epoch 19, `loss/cov` was 1.998 (VICReg's covariance loss is close to 2 at its ceiling for this weighting) and `stats/avg_offdiag_corr_sq` was 0.9995. The features were almost completely redundant across dimensions.

`VICRegWeights` weights align and var 25x against cov at 1x, so the total loss drops fast once alignment and the variance floor are satisfied, which happens within about 15-20 epochs. Decorrelation, probably the term that matters most downstream, barely shows up in the total-loss trajectory this early. So `monitor="loss"` is a poor proxy for representation quality early in VICReg training. Even a correctly checkpointed encoder here is badly undertrained.

### Results

| Encoder snapshot | Source | Linear probe (10 epochs, AdamW) | kNN (k=200, T=0.1) |
|---|---|---|---|
| `vicreg_encoder.weights.h5` | as shipped, epoch 100 (post-collapse) | **10.00%** (chance) | 23.86% |
| `vicreg_encoder_bestckpt.weights.h5` | corrected, epoch 19 (best loss, still undertrained) | **17.92%** | 26.84% |
| README's reported baseline | epoch 100, batch 512, different hardware | 51.95% | 46.39% (best of a k/T grid) |

The linear probe and kNN disagree on the same as-shipped encoder (10% against 23.9%). kNN uses feature-space distances and is scale-invariant, while the linear head is trained with AdamW at a fixed lr=0.003. With `avg_std` probably still far from 1.0 after only partly recovering from the epoch-25 explosion, the optimizer may simply be badly scaled for these features, on top of the features being poor. This isn't fully disentangled.

Raw CSVs: `results/pretrain-c10_baseline/{linear_eval_final, linear_eval_bestckpt, knn_eval_final, knn_eval_bestckpt}.csv`.

## Fix for finding 2 (checkpoint reload)

Applied in `scripts/train_vicreg.py`: `safe_load_trainer_weights(trainer, full_ckpt)` now runs right before the final `encoder.save_weights(...)`. `vicreg_encoder.weights.h5` therefore always reflects the best checkpoint that `ModelCheckpoint` saved, not the state in memory when `fit()` returns. Verified by rerunning the same recipe (below).

## Run 2: `pretrain-c10_baseline_fixed_20260908-1450` (baseline VICReg, fix verification)

- **Purpose:** confirm the finding-2 fix, and check whether finding 1 reproduces.
- **Command:** identical to run 1, with `--run-name pretrain-c10_baseline_fixed`.
- **Output:** `checkpoints_tf/pretrain-c10_baseline_fixed_20260908-1450/`.
- **Fix confirmed:** the logs show `[utils] Loading weights: .../vicreg_full.weights.h5` and `Weights loaded (full structural match)` right before `[train] Done.`. The encoder snapshot now comes from the best checkpoint (epoch 32, loss 4.46), not from final-epoch memory.

### Finding 4: the instability reproduces, at a different epoch

Same blowup as run 1, this time at epoch 33 to 34. `avg_std` reaches 260.9 at epoch 33 and `loss/align` explodes to 9,908 (total loss 247,705) at epoch 34. It happened in both independent runs at slightly different points, which fits data-shuffling randomness changing *when* it triggers but not *whether*. That is strong evidence it is a repeatable property of this recipe at this batch size and hardware, not a fluke.

### Finding 5: recovery was better this time, but best-by-loss can't see it and it can't be checked now

Run 1 plateaued near loss 35. This run recovered further: total loss reached 7.29 at epoch 100 and was still falling. `stats/avg_offdiag_corr_sq` at epoch 100 was **0.928**, clearly more decorrelated than the pre-spike epoch-32 checkpoint's **0.9994**. So the instability, while catastrophic for the loss curve, may have knocked the model out of the redundant-features plateau from finding 3, and further training may have produced a better encoder than the one `ModelCheckpoint` kept.

This can't be verified. `ModelCheckpoint(save_best_only=True)` only writes the best-loss weights, so the epoch-100 weights were never saved and vanished when the container exited. The fix for finding 2 correctly ships the best-loss checkpoint, but it leaves no way to compare it with the final epoch afterward. A second, unconditional checkpoint (for example `vicreg_full_last.weights.h5`, saved every epoch or at the end) would keep both endpoints available.

### Results (updated)

| Encoder snapshot | Run | Linear probe (10 epochs, AdamW) | kNN (k=200, T=0.1) |
|---|---|---|---|
| `vicreg_encoder.weights.h5` | run 1, as shipped, epoch 100 (bug: post-collapse) | 10.00% (chance) | 23.86% |
| `vicreg_encoder_bestckpt.weights.h5` | run 1, corrected, epoch 19 (best loss) | 17.92% | 26.84% |
| `vicreg_encoder.weights.h5` | **run 2 (fixed script)**, epoch 32 (best loss) | **21.74%** | **30.35%** |
| README's reported baseline | epoch 100, batch 512, different hardware | 51.95% | 46.39% (best of a k/T grid) |

Run 2 (fix applied, later best epoch, no bug compounding it) is the best result so far but is still well below the README's numbers. That is consistent with finding 3 (the best-by-loss checkpoints this recipe picks are undertrained on decorrelation) plus the batch size 256 against 512 difference. Raw CSVs: `results/pretrain-c10_baseline/{linear_eval_fixed, knn_eval_fixed}.csv`.

## Screening study (2026-09-19)

Goal: find out why the baseline is unstable and whether anything in the recipe explains the low accuracy, before comparing baseline with `--adaptive`. Every run used the current code in the NGC container, batch size 256, `--seed 0`, `--use-schedules`, and a 100-epoch schedule stopped at epoch 40 with `--stop-epoch 40`. Unless noted, `--lr 0.01` and `--w-cov 1`, as in runs 1 and 2. kNN is k=200, T=0.1 on the epoch-40 encoder. "Stable" was fixed before the runs: after epoch 3, `avg_std` never above 5 and the epoch loss never more than 3x the previous epoch. Runs are in `checkpoints_tf/screen-*`.

| Arm | Change from the default | Outcome | `corr^2` at the end | kNN top-1 |
|---|---|---|---|---|
| control | none | stable by the criterion, but collapsed | 1.0000 | 28.19% |
| warmup5 | `--warmup-epochs 5` | collapsed by epoch 17, NaN at epoch 19 | 1.0000 | n/a |
| clip1 | `--clipnorm 1.0` | NaN in epoch 2 | n/a | n/a |
| lr003 | `--lr 0.003` | stable, no spikes (max `loss/align` 0.59), collapsed | 0.9991 | 29.12% |
| wcov4095 | `--w-cov 4095` | unstable: epoch-1 spike, 4.2x loss jump at epoch 22, `avg_std` drifts to 0.65 | 0.0274 | 25.48% |
| wcov4095-warmup5 | `--w-cov 4095 --warmup-epochs 5` | unstable: 12.3x jump at epoch 4, `avg_std` up to 227, then settles | 0.0182 | 27.18% |
| **wcov4095-lr003-warmup5** | `--w-cov 4095 --lr 0.003 --warmup-epochs 5` | **stable** (max `loss/align` 0.77, worst jump 1.0x, `avg_std` settles at 1.05) | **0.0059** | **53.63%** |

`corr^2` is `stats/avg_offdiag_corr_sq` on the projector output. The criterion was checked against runs 1 and 2 first: it flags both (epoch 25 and epoch 34).

### Finding 6: the covariance term is about D times weaker than the paper's

`covariance_loss` is the mean of squared off-diagonal correlations over D(D-1) entries. VICReg sums the squared covariances and divides by D, so the same weight of 1 is about D-1 times stronger in the paper (4095 for `--proj-out 4096`). On synthetic correlated embeddings the ratio is 81, 611 and 4,639 at D = 64, 512 and 4096. At weight 1 the term is negligible against invariance and variance. That fits every earlier run: `loss/cov` stuck near 2 (its maximum) and `avg_offdiag_corr_sq` stuck at 1.0, so the projector output was rank 1. It also fits finding 3. `--w-cov 4095` is now available as a flag; `--w-cov 1` stays the default so old runs reproduce.

### Finding 7: neither fix works alone

Only the covariance fix (wcov4095) decorrelates the projector but adds spikes and lowers kNN accuracy. Only the lower LR (lr003) is calm but stays collapsed. Together with a 5-epoch warmup they give a stable, decorrelated run, and the encoder reaches 53.63% kNN at epoch 40, against 25 to 29% for every other arm. That already beats the 46.39% (best over a grid) that the README reported for a finished 100-epoch run. Caveat: one seed, half a schedule, and batch size 256 here against 512 there is still an unresolved confound in that comparison.

### Finding 8: a rank-1 projector doesn't make the encoder useless

The control's projector was fully collapsed and its encoder still reached 28% kNN. The eval scripts read the 2048-d encoder features, not the projector output. So `corr^2` on the projector is a health signal, not an accuracy predictor. The covariance fix alone did not raise accuracy either (25.48%).

### Finding 9: the epoch-1 spike was always there

The epoch-1 mean loss was 4.34e3 in run 1 and 5.92e3 in run 2 (from their `history.jsonl`), and 3.22e3 in the control here, against a best loss of about 4.5. With LR 0.01 and no warmup, `loss/align` explodes within the first two steps (358,569 mean loss at step 2 in a 2-epoch check). With LR 0.003 and a 5-epoch warmup the epoch-1 loss was 157.

### Finding 10: two runs went NaN, and the cause is not isolated

`warmup5` first collapsed (`avg_std` 0.70 then 0.56, `corr^2` 1.000 at epochs 17 and 18) and then produced NaN at epoch 19, batch 74. `clip1` produced NaN in epoch 2, batch 141, right after a 4,319 epoch-1 loss. Gradient clipping did not prevent it. One untested hypothesis: `variance_loss` uses `tf.math.reduce_std` with no epsilon, so a constant embedding dimension gives a non-finite gradient, whereas reference VICReg uses `sqrt(var + 1e-4)`. The recipe that avoids the collapse also avoided NaN in all 40 epochs, so this was not pursued.

## Final comparison: baseline vs `--adaptive` (2026-09-19)

Recipe from the screening study: batch 256, `--use-schedules --warmup-epochs 5 --lr 0.003 --w-cov 4095`, 100 epochs. Two runs at `--seed 0`: baseline (no extra flags) and `--adaptive` (only that flag, not `--adaptive-targets`, so the reweighter is the one variable). Evaluated both the best-loss and last-epoch encoders (linear probe, 10 epochs AdamW; kNN, k=200, T=0.1). Runs are `checkpoints_tf/final-baseline-s0_20260919-1640` and `checkpoints_tf/final-adaptive-s0_20260919-1732`.

| Method | Encoder | Linear top-1 | kNN top-1 |
|---|---|---|---|
| VICReg (baseline) | best-loss (epoch 96) | **61.44%** | **56.52%** |
| VICReg (baseline) | last (epoch 100) | 61.26% | 56.55% |
| AdaptiveVICReg | best-loss (epoch 96) | 31.04% | 35.59% |
| AdaptiveVICReg | last (epoch 100) | 31.01% | 35.35% |

Baseline nearly doubles `--adaptive` on both metrics, and both baseline numbers clear the README's historical 51.95% linear / 46.39% kNN (batch 512, different hardware, old recipe). Best-loss and last-epoch agree with each other to within 0.3 points for both methods, so at this recipe the choice of checkpoint no longer matters much, unlike in runs 1 and 2.

### Finding 11: `--adaptive` collapses variance regardless of `--w-cov`

At epoch 100 the adaptive run's logged weights were `w_sim=125.00` (5.00x `w0.sim`, the loss-magnitude clip's ceiling), `w_cov=20650.78` (5.04x `w0.cov`, also at the ceiling once the embedding-health boost is folded in), and `w_var=23.81` (0.95x `w0.var`, just under the constant-weight baseline). `stats/avg_std` stayed at 0.093 for the whole run (baseline: 1.142), so the projection never left its collapsed state even though `--adaptive` exists specifically to react to collapse. Oddly, `avg_offdiag_corr_sq` was even lower than baseline's (0.0043 vs 0.0053): the huge `w_cov` did decorrelate the (collapsed) projection, it just didn't help the encoder.

Mechanism: `AdaptiveReweighter`'s magnitude balancing sets each term's multiplier to `mean(EMAs) / that term's EMA`, clipped to `[0.2, 5.0]`. Once std is collapsed, `l_var` (bounded above by `relu(gamma - std)` near `gamma=1`) is naturally much larger than `l_align` and `l_cov` (both near zero once alignment is satisfied and correlations are small), so magnitude balancing alone pushes `w_var` down and `w_sim`/`w_cov` up to the 5x ceiling, whatever `w0` is. The embedding-health boost is meant to counter this for variance, but it multiplies the already-shrunk `mag_var` and is itself capped at 4x, so in the worst case it can only bring `w_var` back to roughly baseline, never above it. The two terms that are already satisfied get 5x more weight, and the one term that needs fixing gets no more than it would without any reweighting. That is a fixed point: once collapsed, there is no path back.

This does not depend on the covariance-scale fix in finding 6. Feeding the reweighter magnitudes representative of this collapsed regime (`l_align` and `l_cov` near zero, `l_var` near its ceiling of about 2, the shape both runs show in their first few epochs) at both `w0.cov=1` and `w0.cov=4095` gives the same weight ratios in both cases (`w_sim` 5.00x; `w_var` 0.35x with a healthy probe, 0.99x with a collapsed one; `w_cov` 5.03-5.04x). Only the absolute `w_cov` scales with `w0.cov`; the ratios, and so the failure mode, are unchanged. This is a property of the reweighter itself, not an interaction with the `--w-cov` fix.

Caveat: baseline used one seed here (seed 0). A second baseline seed turned out to matter a great deal; see finding 12. A second `--adaptive` seed was not run. Given the mechanism above is close to deterministic once the embedding collapses, a second seed is unlikely to change the direction of that result, but it hasn't been checked.

### Finding 12: the fixed recipe reduces the instability, it doesn't eliminate it

A second baseline seed (`--seed 1`) was queued for a noise estimate. The container running it was interrupted after epoch 62 (host issue, not a code bug: `vicreg_full.weights.h5` was mid-overwrite and came out corrupted, while the separately-written `vicreg_full_last.weights.h5` from the same epoch was intact and unaffected). It was resumed from there with `resume_pretrain.py` to epoch 100 (`checkpoints_tf/final-baseline-s1_resumed`), which continued smoothly with no discontinuity at the resume point (loss 75.44 at epoch 61, 74.83 at epoch 63, declining steadily to 64.59 at epoch 100).

But this seed was already in trouble long before the resume. Comparing its logged history against seed 0's, at epoch 10 both were similar (`l_var` 0.91 vs 0.68, `avg_std` 0.55 vs 0.66). At **epoch 14**, seed 1's loss jumped from 78.7 to 858.1 and `avg_std` spiked to 210.5, the same kind of event as findings 1 and 4, just smaller (finding 1's spike reached 14,767). By epoch 15 `avg_std` had already crashed back down to 0.96 and `l_var` to 0.047, and it never recovered from there: for the rest of training (including after the resume, through epoch 100) `l_var` stayed between 0.89 and 0.27 and `avg_std` stayed between 0.88 and 0.54, never approaching seed 0's 0.04 / 1.1.

Downstream, this seed reached only 37.0% linear and 34.9% kNN top-1 (best-loss checkpoint), against seed 0's 61.4% / 56.5%, a result almost as far below seed 0's baseline as `--adaptive` was in finding 11.

| Method | Seed | Encoder | Linear top-1 | kNN top-1 |
|---|---|---|---|---|
| VICReg (baseline) | 0 | best-loss | 61.44% | 56.52% |
| VICReg (baseline) | 1 | best-loss (post-resume) | 37.0% | 34.88% |
| AdaptiveVICReg | 0 | best-loss | 31.04% | 35.59% |

This means the fixed recipe (finding 7) makes the instability smaller and less frequent, not gone: it still fired for one of the two seeds tried, at about 1/70th the severity (`avg_std` peaked at 210.5, not 14,767) and without diverging to NaN. A single stable-looking seed is not enough to trust a recipe. The comparison in finding 11 used only seed 0 for baseline; that comparison's *mechanism* is independently verified (the CPU-only synthetic test), so it does not depend on this seed being representative, but the specific 56.5% number might be closer to a lucky draw than a typical one.

One more thing surfaced while resuming: `resume_pretrain.py` calls `safe_load_trainer_weights` before `trainer.compile()`, so the freshly constructed AdamW optimizer never receives the saved momentum and variance state; a resumed run's optimizer restarts cold at whatever step it resumes from, even though the LR schedule correctly continues from that step. The loss trajectory is smooth across the resume point, so it caused no visible blowup, but a smooth loss curve can't rule out a subtler effect on where the run eventually converges, and there is no non-resumed control here to check against. A "resumed" run is not quite the same as an uninterrupted one, and it would be worth fixing (compile before loading, or load into an already-compiled trainer) before relying on resumed runs for results that need to match a continuous run exactly.

## Five-seed baseline sweep (2026-09-30)

Three more baseline seeds (2, 3, 4) at the same recipe as finding 12 (`--lr 0.003 --warmup-epochs 5 --w-cov 4095`, batch 256, 100 epochs), to get an actual failure-rate estimate instead of one extra data point. Runs are `checkpoints_tf/baseline-s{2,3,4}_*`.

| Seed | Instability event | Encoder | Linear top-1 | kNN top-1 |
|---|---|---|---|---|
| 0 | none | best-loss | 61.44% | 56.52% |
| 1 | epoch 14, `avg_std` to 210.5, never recovered | best-loss (post-resume) | 37.0% | 34.88% |
| 2 | none | best-loss | **62.42%** | **59.64%** |
| 2 | none | last (epoch 100) | 62.74% | 59.59% |
| 3 | epoch 14, `avg_std` to 3.05, partial recovery | best-loss | 44.11% | 38.52% |
| 3 | epoch 14, `avg_std` to 3.05, partial recovery | last (epoch 100) | 28.79% | 30.64% |
| 4 | epoch 17, `avg_std` to 3.08, partial recovery | best-loss | 44.78% | 40.04% |
| 4 | epoch 17, `avg_std` to 3.08, partial recovery | last (epoch 100) | 26.63% | 27.52% |

### Finding 13: the instability hits most runs, at wildly different severities, and isn't tied to a specific epoch

Three of five seeds (1, 3, 4) hit an instability event; only seeds 0 and 2 stayed clean. That is a 60% failure rate on five runs, not the occasional exception finding 12 left open.

Seeds 1 and 3 both spiked at epoch 14, which looked like it might be schedule-related, but seed 4 spiked at epoch 17. Nothing in `callbacks.py` or `schedules.py` keys any behavior to a specific epoch once the 5-epoch warmup ends, so the epoch-14 repeat was coincidence, not a schedule boundary.

Severity varies by close to two orders of magnitude. Seed 1's spike reached `avg_std` 210.5 (peak `loss/align` 9.06 was not logged for seed 1's spike epoch itself, only `avg_std` and total loss). Seeds 3 and 4 peaked at `avg_std` 3.05 and 3.08, about 1/69th of seed 1's peak and roughly 1/4,800th of finding 1's original 14,767 spike, yet still large enough to leave a lasting mark: in both seed 3 and seed 4, `l_var` (0.87-0.86 pre-spike) settled into a plateau around 1.06-1.11 for the rest of training instead of continuing to decay toward the near-zero values seeds 0 and 2 reach, and `avg_std` settled around 0.44-0.47 instead of climbing toward 1.05. The recovery is real (loss and `avg_std` both drop sharply within one to two epochs of the peak) but incomplete, and the model never gets back on the healthy trajectory.

This also reopens what finding 12 observed about best-loss and last-epoch agreeing once the recipe was fixed. That held for seed 0 (within 0.3 points) and again for seed 2 here (within 0.3-0.4 points), but not for the two seeds that partially recovered: seed 3 dropped from 38.52% kNN (best-loss) to 30.64% (last), and seed 4 from 40.04% to 27.52%. Once a run partially recovers rather than staying clean or collapsing outright, it keeps drifting after the point `ModelCheckpoint` calls best, so the two snapshots diverge again.

The underlying cause is still the open question from findings 1, 4 and 10. Five runs is enough to say the fixed recipe is unstable more often than not, not enough to say why, or to rule out that batch size 256 (against the paper's much larger batches) is itself part of the cause.

## Fix for finding 11 (`AdaptiveReweighter` variance floor)

Applied in `schedules.py`: magnitude balancing's clip range for the variance multiplier is now `[1.0, 5.0]` (a separate `var_mag_clip`, overridable), instead of sharing `[0.2, 5.0]` with sim and cov. So this signal can raise `w_var` above `w0.var` but never lower it below, which was the specific fixed point finding 11 describes (a collapsed embedding makes `l_var` the dominant raw term, and the old code read that as a reason to lower its weight). Sim and cov keep the original `[0.2, 5.0]` range. Verified with new unit tests in `tests/test_schedules.py` that feed the reweighter the same collapsed-regime magnitudes finding 11's synthetic test used, and check `w_var` never drops below `w0.var`.

This closes one path into the fixed point, not the mechanism itself: magnitude balancing still weighs each term by its raw loss value, not by its actual contribution to the gradient. It hasn't been re-run yet; a second `--adaptive` seed with the fix applied is next, to see whether it actually prevents (or just delays) the collapse in a real run rather than only in the synthetic test.

## Fix for finding 12 (resume optimizer state)

Applied in `resume_pretrain.py`: the optimizer is now compiled and its slot variables explicitly built (`opt.build(encoder.trainable_variables + projector.trainable_variables)`) before `safe_load_trainer_weights` runs, not after. Previously `trainer.compile(optimizer=opt)` ran after the load call, so at load time `trainer` had no `optimizer` attribute at all, meaning none of the checkpoint's saved momentum and variance state could ever have been restored.

This is confirmed directly, not just reasoned about: loading into a compiled-but-unbuilt optimizer (the old order) makes Keras print its own warning that it is skipping the optimizer's variables because the counts don't match, while building the slots first restores every one of them to the exact saved value (`tests/test_model.py::test_optimizer_state_survives_checkpoint_when_built_before_load` and the paired negative-case test). Finding 12's resumed run was affected by this: its optimizer did restart cold at the resume point. The loss curve stayed smooth there regardless, so this does not change that run's read, only removes the open question of why the state wasn't restored.

## Fix for finding 10 (partial): a `--var-eps` flag

### Finding 14: `reduce_std`'s gradient is undefined at exact collapse

A direct test (`tests/test_losses.py::test_variance_loss_default_gradient_is_not_finite_at_full_collapse`) confirms `tf.math.reduce_std`'s gradient is non-finite when every embedding dimension has std exactly 0, while `sqrt(var + 1e-4)` gives a finite gradient in the same state. This is half of finding 10's untested hypothesis, now confirmed at the math level: both of finding 10's NaN runs (`warmup5`, `clip1`) reached full collapse (`corr^2` 1.0) shortly before going NaN, so a non-finite variance-loss gradient at that point is a plausible mechanism.

It does not by itself explain the large-but-finite spikes in findings 1, 4, 12 and 13, since those never produced NaN and a real embedding landing at exactly 0 std is unlikely. A near-zero (not exactly zero) std could still produce a very large, merely finite, gradient through the same term, which an epsilon should also dampen, but that is a different claim and hasn't been checked.

`variance_loss` now takes an optional `eps` (default 0, the original formula, bit-identical to before), threaded through as `--var-eps` in `train_vicreg.py` and `resume_pretrain.py`. No run has used a nonzero value yet.

## Fix verification and ablations (2026-09-30)

Eight runs at the fixed recipe (`--lr 0.003 --warmup-epochs 5 --w-cov 4095 --use-schedules`, 100 epochs unless noted), testing the two fixes above and three open ablations from `TASKS.md`. Runs are `checkpoints_tf/{adaptive-fixed,targets,batch128,vareps}-*`.

| Run | Seed | Linear (best) | kNN (best) | Linear (last) | kNN (last) |
|---|---|---|---|---|---|
| `--adaptive`, fixed reweighter | 0 | 46.43% | 45.22% | 46.87% | 45.23% |
| `--adaptive`, fixed reweighter | 1 | 41.82% | 38.82% | 41.65% | 38.84% |
| `--adaptive-targets` | 0 | 45.57% | 38.54% | **58.82%** | **53.44%** |
| `--adaptive-targets` | 1 | 43.99% | 36.75% | **57.05%** | **51.88%** |
| Baseline, batch 128 | 0 | 55.78% | **60.02%** | 55.79% | 60.01% |
| Baseline, batch 128 | 1 | 34.53% | 35.51% | 35.06% | 35.40% |
| Baseline, `--var-eps 1e-4`, seed 3's recipe, stopped at epoch 30 | 3 | 46.10% | 40.77% | 24.50% | 25.93% |
| Baseline, `--var-eps 1e-4`, seed 4's recipe, stopped at epoch 30 | 4 | 45.68% | 40.57% | 26.17% | 29.49% |

### Finding 15: the reweighter fix ends the collapse, but `--adaptive` still trails baseline

Both fixed-reweighter seeds stayed healthy for the entire run: `avg_std` climbed past 1.0 by epoch 10 and held there (seed 0: 1.068 at epoch 100; seed 1: 1.125), `l_var` settled near 0 (0.003 and 0.019), nothing like finding 11's `avg_std` 0.093 for the whole run. Seed 0 improved from the broken reweighter's 31.04% / 35.59% to 46.43% / 45.22%, a real fix. Seed 1, run at the same seed where *plain baseline* catastrophically failed (finding 13's 37.0% / 34.9%), actually beat that baseline number (41.82% / 38.82%) while staying collapse-free itself.

But neither fixed-reweighter seed approaches a clean baseline run's 61-62% linear / 56-60% kNN. Looking at `loss/w_sim` over the run: magnitude balancing pushes it down to about 8.4 (0.34x `w0.sim`) for most of training and keeps it there, once `l_align` becomes large relative to the now-healthy `l_var` and `l_cov`. The fix closes the specific trap in finding 11 (a collapsed embedding no longer gets its variance weight lowered), but the same raw-loss-value balancing now suppresses the invariance term instead, once that's the largest raw loss. Fixing one term's floor didn't fix the underlying design: the reweighter still treats "largest raw loss" as "this term needs less weight," which is backwards whenever that term is the one the model hasn't finished optimizing rather than one that's already collapsed.

### Finding 16: `--adaptive-targets`'s decaying `nu` makes "best loss" pick the least-decorrelated checkpoint

Both `--adaptive-targets` seeds show a large best-loss-vs-last-epoch gap (about 13 and 13 points linear, 15 and 15 points kNN), much bigger than any clean baseline run's gap. The cause is in `targets-s0`'s own history: at epoch 6, loss is 1.95 (the run's lowest) with `avg_offdiag_corr_sq` at **0.989**, a nearly fully redundant projector. By epoch 100, loss has risen to 49.8, its highest point in the run, with `avg_offdiag_corr_sq` down to **0.0055**, a well-decorrelated one. `nu` (the correlation target) decays from 1.0 to 0.0 over training, so `l_cov = mean((corr - nu)^2)` stays near zero while `nu` is still near 1 and correlations are still near 1, then grows as the model is pushed toward genuine decorrelation and `nu` falls. Loss goes up exactly as the representation gets better. `ModelCheckpoint(monitor="loss")` has no way to know this and picks the early, redundant epoch every time.

Read from the last epoch instead, `--adaptive-targets` performs close to a clean baseline run (58.82% / 53.44% and 57.05% / 51.88%, against baseline's 61-62% / 56-60%), which is a much more favorable comparison than the best-loss numbers suggested. This is the same failure mode `TASKS.md`'s "Checkpointing" item already names in general (loss isn't a held-out metric), but here it isn't just an early-training artifact (finding 3): the schedule makes loss systematically anti-correlated with representation quality for the entire run.

### Finding 17: batch size 128 still fails for one of two seeds, with a different shape than the spike

Seed 0 at batch 128 ran cleanly and reached the best kNN top-1 of any run so far (60.02%, best-loss and last-epoch agreeing to 0.01 points). Seed 1 did not: instead of a sharp one-epoch spike like findings 1, 4, 12 and 13, `avg_std` declined gradually from 0.81 (epoch 10) to 0.42 (epoch 25) while `l_var` rose gradually from 0.62 to 1.09 over the same stretch, no single epoch standing out the way loss jumps by 10x+ in the spike cases. It partially recovered by epoch 100 (`avg_std` 0.91, `l_var` 0.33) but landed at 34.53-35.51%, in the same range as the other failed runs. Two seeds is nowhere near enough to say whether batch size changes the failure rate, but it is enough to say a smaller batch doesn't prevent failure, and that failure can look qualitatively different (a slow drift instead of a spike-and-partial-recovery).

### Finding 18: `--var-eps` does not prevent the spikes it wasn't built for

Re-running seed 3's and seed 4's exact recipes and seeds with `--var-eps 1e-4` added, both spiked again at the identical epoch (14), with `loss/align` reaching 4.21 and 13.90 (against the original runs' 9.06 and 9.89, so the epsilon didn't consistently reduce severity either). This confirms finding 14's caveat: `--var-eps` fixes the non-finite gradient at *exact* collapse (relevant to finding 10's two NaN runs), but the large-but-finite spikes in findings 1, 4, 12, 13 and 17 have a different, still-unidentified cause. The root cause of the instability itself remains open.

## Next steps

1. **Done: unconditional last-epoch checkpoint.**
2. **Done: the instability, partially.** LR 0.003, a 5-epoch warmup and `--w-cov 4095` together were the only stable, decorrelated setup in the screening study, but a 5-seed sweep shows the fix reduces severity, not frequency: 3 of 5 seeds still hit an instability event (finding 13). The NaN hypothesis in finding 10 is still untested.
3. **Done: baseline vs `--adaptive` at the fixed recipe (seed 0).** `--adaptive` fails for a specific, reproducible reason (finding 11) that does not depend on which seed hit it, verified with a CPU-only synthetic test. Whether seed 0's *baseline* number (56.5% kNN) is typical is less certain: see findings 12 and 13.
4. **Done: a five-seed baseline sweep.** 3 of 5 seeds hit an instability event, at severities ranging over two orders of magnitude (finding 13). The instability itself (findings 1, 4, 10, 12, 13) is still not understood well enough to prevent, only to describe. A second `--adaptive` seed is still needed to know if its failure mode is as consistent as the mechanism in finding 11 suggests.
5. **Done: `AdaptiveReweighter`'s variance floor.** Two seeds confirm it ends the collapse (finding 15). `--adaptive` still trails baseline, now because magnitude balancing suppresses the invariance weight instead, once that's the largest raw term. The reweighter's core design (weighing terms by raw loss value) still needs a rethink, not just another floor.
6. **Done: `resume_pretrain.py`'s optimizer reset.** Verified with a checkpoint round-trip test (`tests/test_model.py`). No real resumed run has re-tested this against the original, but the mechanism is directly confirmed, not just reasoned about.
7. **Done: `--adaptive-targets` ablation.** Its apparent underperformance was mostly a checkpoint-selection artifact: the decaying `nu` makes loss rise as the representation genuinely improves, so best-loss picks a nearly redundant early epoch. Read from the last epoch, it performs close to baseline (finding 16).
8. **Done: `--var-eps 1e-4` against two recipes known to spike.** It did not prevent either spike (finding 18). The epsilon fixes finding 10's NaN cases, not the large-but-finite spikes; their cause is still open.
9. **Partially done: batch size 128 as an instability variable.** One of two seeds failed, with a different (gradual, not spiky) failure shape (finding 17). Two seeds isn't enough to compare failure rates against batch 256's five-seed sample.
10. **Open: the instability's root cause.** Still unexplained after 9 baseline-recipe seeds across two batch sizes and one ruled-out hypothesis (finding 18). Nothing tried so far (warmup, covariance scale, batch size, an epsilon) has stopped it, only sometimes changed its shape or severity.
11. **Open: redesign `AdaptiveReweighter`'s magnitude balancing.** It weighs each term by its raw loss value, not by whether that term still needs optimizing. Finding 15 shows this keeps causing new failure modes (invariance suppression) even after the specific variance-collapse trap is closed.
12. **Decide the default for `--w-cov`.** It defaults to 1, which finding 6 shows is about D times weaker than the paper's scale. Changing the default improves results but changes what "baseline" means for anyone already using this repo.
13. **Feed the runs into `src/vicreg_tf/report_metrics.py`** for the comparison plots and tables described in the README.
