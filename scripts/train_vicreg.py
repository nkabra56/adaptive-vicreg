"""Pretrain a VICReg encoder, optionally with the adaptive mechanisms.

`--adaptive` reweights the loss terms every step and `--adaptive-targets` schedules the gamma/nu targets.
With neither flag this is baseline VICReg. The README has example commands.

By default a run writes to `<model-dir>/<run-name>_<timestamp>/`:
  vicreg_full.weights.h5          best-loss weights (encoder, projector, optimizer)
  vicreg_encoder.weights.h5       encoder-only weights taken from that same checkpoint, used by the eval scripts
  vicreg_full_last.weights.h5     trainer weights after the most recent epoch
  vicreg_encoder_last.weights.h5  encoder-only weights at the end of training
  train_config.json               hyperparameters
  metrics/history.jsonl           one JSON record per epoch, read by report_metrics.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    CosineScheduleCallback,
    VicRegMetricsLogger,
    VICRegTrainer,
    VICRegWeights,
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
    p = argparse.ArgumentParser(description="Pretrain a VICReg encoder, optionally with adaptive loss weighting.")
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width.")
    p.add_argument("--proj-out", type=int, default=4096, help="Projector output width.")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector depth (1, 2 or 3).")
    p.add_argument("--lr", type=float, default=0.01, help="Base learning rate.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay.")
    p.add_argument("--w-sim", type=float, default=25.0, help="Base weight of the invariance term.")
    p.add_argument("--w-var", type=float, default=25.0, help="Base weight of the variance term.")
    p.add_argument("--w-cov", type=float, default=1.0,
                   help="Base weight of the covariance term. The paper sums squared covariances and divides "
                        "by the embedding width, so its weight of 1 matches about --proj-out minus 1 here.")
    p.add_argument("--adaptive", action="store_true",
                   help="Reweight the loss terms every step (Adaptive VICReg). Omit for constant weights.")
    p.add_argument("--adaptive-targets", action="store_true",
                   help="Ramp the variance floor gamma and correlation target nu during training instead of "
                        "using 1.0 and 0.0. Can be combined with --adaptive.")
    p.add_argument("--ema-decay", type=float, default=0.98, help="EMA decay used by --adaptive.")
    p.add_argument("--var-boost-k", type=float, default=2.0,
                   help="Boost strength for the variance weight when the std drops below target (--adaptive only).")
    p.add_argument("--cov-boost-k", type=float, default=2.0,
                   help="Boost strength for the covariance weight when off-diagonal correlation rises above "
                        "target (--adaptive only).")
    p.add_argument("--use-schedules", action="store_true", help="Apply cosine LR and weight-decay schedules per step.")
    p.add_argument("--warmup-epochs", type=float, default=0.0,
                   help="Ramp the LR up linearly over this many epochs before the cosine decay. "
                        "Needs --use-schedules.")
    p.add_argument("--clipnorm", type=float, default=0.0,
                   help="Clip each gradient tensor to this norm. 0 turns clipping off.")
    p.add_argument("--stop-epoch", type=int, default=None,
                   help="Stop after this epoch but keep the LR schedule sized for --epochs. For screening runs.")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf", help="Root output directory.")
    p.add_argument("--run-name", type=str, default=None, help="Run folder prefix. A timestamp is appended.")
    p.add_argument("--ckpt-out", type=str, default=None,
                   help="Explicit path for the full weights file. Overrides --model-dir.")
    p.add_argument("--metrics-probe-batch", type=int, default=256, help="Size of the fixed probe batch for stats.")
    p.add_argument("--metrics-compute-on", choices=["projector", "encoder"], default="projector",
                   help="Where to compute embedding stats.")
    p.add_argument("--record-every", type=int, default=1, help="Record metrics every N epochs.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed Python, NumPy and TF. Omit for a nondeterministic run.")
    add_device_arg(p)
    args = p.parse_args()
    if args.warmup_epochs and not args.use_schedules:
        p.error("--warmup-epochs needs --use-schedules")
    if args.stop_epoch is not None and not 1 <= args.stop_epoch <= args.epochs:
        p.error("--stop-epoch must be between 1 and --epochs")
    return args


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_global_seed(args.seed)
    set_mixed_precision(False)  # float32 for stability with VICReg

    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(
        f"[train] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
        f"epochs={args.epochs} steps/epoch={steps_per_epoch}"
    )

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M")
    if args.ckpt_out:
        os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
        out_dir = os.path.dirname(args.ckpt_out)
        full_ckpt = args.ckpt_out
        enc_ckpt = full_ckpt.replace(".weights.h5", ".encoder.weights.h5")
    else:
        run_prefix = args.run_name or "pretrain"
        out_dir = os.path.join(args.model_dir, f"{run_prefix}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        full_ckpt = os.path.join(out_dir, "vicreg_full.weights.h5")
        enc_ckpt = os.path.join(out_dir, "vicreg_encoder.weights.h5")
    last_full_ckpt = full_ckpt.replace(".weights.h5", "_last.weights.h5")
    last_enc_ckpt = enc_ckpt.replace(".weights.h5", "_last.weights.h5")

    with open(os.path.join(out_dir, "train_config.json"), "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "dataset": args.dataset,
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "feat_dim": args.feat_dim,
                "proj_out": args.proj_out,
                "proj_layers": args.proj_layers,
                "lr": args.lr,
                "wd": args.wd,
                "w_sim": args.w_sim,
                "w_var": args.w_var,
                "w_cov": args.w_cov,
                "adaptive": args.adaptive,
                "adaptive_targets": args.adaptive_targets,
                "ema_decay": args.ema_decay,
                "var_boost_k": args.var_boost_k,
                "cov_boost_k": args.cov_boost_k,
                "use_schedules": args.use_schedules,
                "warmup_epochs": args.warmup_epochs,
                "clipnorm": args.clipnorm,
                "stop_epoch": args.stop_epoch,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    device_str = select_device(args.device, "train")
    print(f"[train] Using device scope: {device_str}")

    with tf.device(device_str):
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)
        opt = keras.optimizers.AdamW(
            learning_rate=args.lr,
            weight_decay=args.wd,
            clipnorm=args.clipnorm if args.clipnorm > 0 else None,
        )

        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=args.w_sim, var=args.w_var, cov=args.w_cov),
            adaptive_weights=args.adaptive,
            adaptive_targets=args.adaptive_targets,
            use_schedules=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            base_lr=args.lr,
            base_wd=args.wd,
            reweighter_kwargs={
                "decay": args.ema_decay,
                "k_std": args.var_boost_k,
                "k_cov": args.cov_boost_k,
            },
        )
        trainer.compile(optimizer=opt)

        force_build_for_saving(trainer, encoder, projector, args.image_size)

        sample_images = take_probe_batch(ds, size=args.metrics_probe_batch)

        cbs: List[keras.callbacks.Callback] = [
            keras.callbacks.ModelCheckpoint(
                filepath=full_ckpt,
                save_weights_only=True,
                monitor="loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(filepath=last_full_ckpt, save_weights_only=True, verbose=0),
            keras.callbacks.TerminateOnNaN(),
            VicRegMetricsLogger(
                run_dir=out_dir,
                encoder=encoder,
                projector=projector,
                sample_images=sample_images,
                compute_on=args.metrics_compute_on,
                record_every=args.record_every,
            ),
        ]
        if args.use_schedules:
            cbs.insert(
                0,
                CosineScheduleCallback(
                    optimizer=opt,
                    total_steps=steps_per_epoch * args.epochs,
                    base_lr=args.lr,
                    base_wd=args.wd,
                    warmup_frac=args.warmup_epochs / args.epochs,
                ),
            )

        trainer.fit(
            ds,
            epochs=args.stop_epoch or args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=cbs,
            verbose=1,
        )

        # The in-memory state is the end of training. Save it before the best checkpoint is reloaded.
        encoder.save_weights(last_enc_ckpt)

        # ModelCheckpoint keeps the best-loss weights on disk. Reload them so the encoder snapshot comes
        # from that checkpoint and not from whatever state the last epoch left in memory.
        safe_load_trainer_weights(trainer, full_ckpt)
        encoder.save_weights(enc_ckpt)

    print(
        f"[train] Done.\n"
        f"  Encoder -> {enc_ckpt}\n"
        f"  Full    -> {full_ckpt}\n"
        f"  Last    -> {last_enc_ckpt}, {last_full_ckpt}\n"
        f"  Config  -> {os.path.join(out_dir, 'train_config.json')}\n"
        f"  Metrics -> {os.path.join(out_dir, 'metrics', 'history.jsonl')}"
    )


if __name__ == "__main__":
    main()
