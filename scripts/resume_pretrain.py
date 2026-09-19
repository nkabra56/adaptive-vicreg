"""Resume VICReg pretraining from a checkpoint.

Rebuilds the encoder, projector and trainer the way `train_vicreg.py` does, loads the weights, and keeps
training with an optional LR warmup and a loss-explosion guard. Metrics go to `<model-dir>/metrics/history.jsonl`.
Pass the same `--adaptive`, `--adaptive-targets` and `--feat-dim` values as the original run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    CosineScheduleCallback,
    LossExplosionGuard,
    VicRegMetricsLogger,
    VICRegTrainer,
    VICRegWeights,
    WarmupLR,
    add_device_arg,
    build_dataset,
    build_encoder,
    build_projector,
    force_build_for_saving,
    preparse_device,
    safe_load_trainer_weights,
    select_device,
    set_global_seed,
    set_mixed_precision,
    steps_for_dataset,
    take_probe_batch,
)

# Has to run before anything initializes CUDA, so it lives at import time.
preparse_device()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume VICReg pretraining from a checkpoint.")
    p.add_argument("--ckpt", type=str, required=True, help="Weights file to load (.weights.h5, full trainer).")
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width (must match the checkpoint).")
    p.add_argument("--proj-out", type=int, default=4096, help="Projector output width.")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector depth (1, 2 or 3).")
    p.add_argument("--lr", type=float, default=0.01, help="Base learning rate.")
    p.add_argument("--resume-lr", type=float, default=None,
                   help="Learning rate to use after resuming. Defaults to --lr.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay.")
    p.add_argument("--w-sim", type=float, default=25.0, help="Base weight of the invariance term.")
    p.add_argument("--w-var", type=float, default=25.0, help="Base weight of the variance term.")
    p.add_argument("--w-cov", type=float, default=1.0,
                   help="Base weight of the covariance term. The paper sums squared covariances and divides "
                        "by the embedding width, so its weight of 1 matches about --proj-out minus 1 here.")
    p.add_argument("--adaptive", action="store_true",
                   help="Adaptive loss weighting. Must match the run being resumed.")
    p.add_argument("--adaptive-targets", action="store_true",
                   help="Gamma/nu target schedule. Must match the run being resumed.")
    p.add_argument("--ema-decay", type=float, default=0.98, help="EMA decay used by --adaptive.")
    p.add_argument("--var-boost-k", type=float, default=2.0, help="Variance-weight boost strength (--adaptive only).")
    p.add_argument("--cov-boost-k", type=float, default=2.0, help="Covariance-weight boost strength (--adaptive only).")
    p.add_argument("--use-schedules", action="store_true",
                   help="Apply the cosine LR and weight-decay schedule, continuing from --initial-epoch. "
                        "The base LR is --resume-lr if given, else --lr.")
    p.add_argument("--initial-epoch", type=int, default=0, help="Epoch the original run stopped at.")
    p.add_argument("--epochs", type=int, default=100, help="Epoch to train up to.")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf/resumed_run", help="Output directory.")
    p.add_argument("--clipnorm", type=float, default=1.0, help="Gradient clip norm. Use 0 to disable.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Ramp the LR up linearly over this many steps after resuming.")
    p.add_argument("--loss-guard", type=float, default=1e12,
                   help="Cut the learning rate 10x when a batch loss exceeds this value.")
    p.add_argument("--metrics-probe-batch", type=int, default=256, help="Size of the fixed probe batch for stats.")
    p.add_argument("--metrics-compute-on", choices=["projector", "encoder"], default="projector",
                   help="Where to compute embedding stats.")
    p.add_argument("--record-every", type=int, default=1, help="Record metrics every N epochs.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed Python, NumPy and TF. Omit for a nondeterministic run.")
    add_device_arg(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_global_seed(args.seed)
    set_mixed_precision(False)

    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(
        f"[resume] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
        f"initial_epoch={args.initial_epoch} final_epochs={args.epochs} steps/epoch={steps_per_epoch}"
    )

    device_str = select_device(args.device, "resume")
    print(f"[resume] Using device scope: {device_str}")

    run_dir = args.model_dir
    os.makedirs(run_dir, exist_ok=True)

    with tf.device(device_str):
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)

        base_lr = args.lr if args.resume_lr is None else float(args.resume_lr)

        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=args.w_sim, var=args.w_var, cov=args.w_cov),
            adaptive_weights=args.adaptive,
            adaptive_targets=args.adaptive_targets,
            use_schedules=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            base_lr=base_lr,
            base_wd=args.wd,
            reweighter_kwargs={
                "decay": args.ema_decay,
                "k_std": args.var_boost_k,
                "k_cov": args.cov_boost_k,
            },
        )

        force_build_for_saving(trainer, encoder, projector, args.image_size)
        safe_load_trainer_weights(trainer, args.ckpt)

        opt = keras.optimizers.AdamW(
            learning_rate=base_lr,
            weight_decay=args.wd,
            clipnorm=args.clipnorm if args.clipnorm > 0 else None,
        )
        trainer.compile(optimizer=opt)

        # curr_step is a tf.Variable. Assigning a plain int would replace it and break train_step's
        # assign_add, so use .assign().
        trainer.curr_step.assign(int(args.initial_epoch) * int(steps_per_epoch))

        ckpt_path = os.path.join(run_dir, "vicreg_tf.weights.h5")

        sample_images = take_probe_batch(ds, size=args.metrics_probe_batch)

        callbacks = []
        if args.use_schedules:
            # Continue the cosine schedule from --initial-epoch. The post-resume warmup scales the schedule.
            callbacks.append(
                CosineScheduleCallback(
                    optimizer=opt,
                    total_steps=steps_per_epoch * args.epochs,
                    base_lr=base_lr,
                    base_wd=args.wd,
                    start_step=args.initial_epoch * steps_per_epoch,
                    ramp_steps=args.warmup_steps,
                )
            )
        else:
            callbacks.append(WarmupLR(base_lr=base_lr, warmup_steps=args.warmup_steps))
        callbacks += [
            LossExplosionGuard(threshold=float(args.loss_guard), factor=0.1),
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path, save_weights_only=True, monitor="loss", save_best_only=True, verbose=1
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(os.path.join(run_dir, "resume_log.csv"), append=True),
            VicRegMetricsLogger(
                run_dir=run_dir,
                encoder=encoder,
                projector=projector,
                sample_images=sample_images,
                compute_on=args.metrics_compute_on,
                record_every=args.record_every,
            ),
        ]

        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )

    print(
        "[resume] Done.\n"
        f"  Best weights -> {ckpt_path}\n"
        f"  Metrics      -> {os.path.join(run_dir, 'metrics', 'history.jsonl')}\n"
        f"  CSV Log      -> {os.path.join(run_dir, 'resume_log.csv')}"
    )


if __name__ == "__main__":
    main()
