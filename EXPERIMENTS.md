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

## Next steps

1. **Add an unconditional final or periodic checkpoint** (finding 5), so the end state of a run can always be compared with best-by-loss.
2. **Investigate the instability** (findings 1 and 4). Try a lower peak LR, a real warmup (`--use-schedules` currently builds `CosineWarmup` with `warmup_frac=0.0`, so there is no warmup at all), or gradient clipping, and see whether it can be prevented or delayed. Finding 5 hints that it may sometimes help by breaking the redundant-features plateau. If so, the better question is how to checkpoint past it, not how to prevent it.
3. **Compare full trajectories.** Once a run's whole trajectory (not just best-by-loss) can be evaluated, compare epoch 100 with best-by-loss, and both with the README's 51.95% / 46.39% baseline. Batch size 256 against 512 remains a confound.
4. **Run `--adaptive` only after there is a well-understood baseline.** Since `main` was merged, `--adaptive` is the loss-weight reweighter, and the old `nu` mechanism is now `--adaptive-targets`. By default that starts `nu` at 1.0 and decays it to 0.0, so for most of training it pushes off-diagonal correlation toward redundancy instead of away from it. That may be a second reason the earlier adaptive results underperformed, so ablate it before trusting a run that uses `--adaptive-targets`.
5. **Feed stable baseline and adaptive runs into `src/vicreg_tf/report_metrics.py`** for the comparison plots and tables described in the README.
