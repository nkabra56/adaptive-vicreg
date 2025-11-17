"""
Script Title: Resume VICReg Pretraining From Existing Weights

Purpose
-------
I use this to safely resume a VICReg run from saved weights. It rebuilds the
encoder+projector+trainer, loads weights, optionally warms up LR, optionally
freezes BN updates for a few steps, guards against loss spikes, and logs the
same per-epoch metrics JSONL as my training script.

Example usage
-------------
python3 scripts/resume_pretrain.py \
  --ckpt checkpoints_tf/pretrain-c10_model9_20251117-1530/vicreg_full.weights.h5 \
  --dataset cifar10 \
  --image-size 32 \
  --batch-size 256 \
  --proj-out 4096 \
  --proj-layers 3 \
  --epochs 200 \
  --initial-epoch 80 \
  --lr 0.01 \
  --resume-lr 0.003 \
  --wd 1e-6 \
  --warmup-steps 500 \
  --bn-freeze-steps 200 \
  --adaptive \
  --use-schedules \
  --record-every 1 \
  --metrics-probe-batch 256 \
  --metrics-compute-on projector \
  --model-dir checkpoints_tf/resumed_run \
  --device auto

How it works (high level)
-------------------------
1) Rebuild encoder/projector/trainer identically to train_vicreg.py.
2) Load weights robustly (full trainer or encoder-only).
3) Optionally apply LR warmup and loss explosion guard callbacks.
4) Log the same losses and embedding stats JSONL for continuity.

Author: Nishant Kabra
Date: 11/17/2025
"""
from __future__ import annotations

# --- Make sure local package "vicreg_tf" (under <repo>/src) is importable. -----
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]   # <repo>
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ------------------------------------------------------------------------------

import argparse
import json
import os
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    VICRegTrainer,
    VICRegWeights,
    build_dataset,
    build_encoder,
    build_projector,
    enable_memory_growth,
    force_build_for_saving,
    gpu_probe_ok,
    print_devices,
    safe_load_trainer_weights,
    set_mixed_precision,
    steps_for_dataset,
)


# =========================== Helper callbacks ==================================

class WarmupLR(keras.callbacks.Callback):
    """
    Linear LR warmup for the first N steps after resume.

    I scale LR from 10% to 100% over `warmup_steps` to avoid sudden jumps.
    """
    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))

    def on_train_batch_begin(self, batch, logs=None):
        if self.warmup_steps <= 0:
            return
        step = int(self.model.curr_step)  # assumes trainer exposes this
        if step < self.warmup_steps:
            scale = 0.1 + 0.9 * (step + 1) / float(self.warmup_steps)
            new_lr = self.base_lr * scale
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)


class LossExplosionGuard(keras.callbacks.Callback):
    """
    Reduce LR if a single batch loss explodes beyond a threshold.
    """
    def __init__(self, threshold: float = 1e8, factor: float = 0.1):
        super().__init__()
        self.threshold = float(threshold)
        self.factor = float(factor)

    def on_train_batch_end(self, batch, logs=None):
        if not logs:
            return
        loss = float(logs.get("loss", 0.0))
        if loss > self.threshold:
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = max(lr * self.factor, 1e-6)
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(f"[loss-guard] loss={loss:.3e} > {self.threshold:.1e} -> lr {lr:.2e} -> {new_lr:.2e}")


class VicRegMetricsLogger(keras.callbacks.Callback):
    """
    Same JSONL logger I use in training: records losses + embedding stats.
    """
    def __init__(self, run_dir: str, encoder: tf.keras.Model,
                 projector: Optional[tf.keras.Model],
                 sample_images: Optional[tf.Tensor],
                 compute_on: str = "projector",
                 loss_keys: Optional[dict] = None,
                 record_every: int = 1) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.encoder = encoder
        self.projector = projector
        self.sample_images = sample_images
        self.compute_on = compute_on
        self.loss_keys = loss_keys or {
            "total": "loss",
            "align": "l_align",
            "var": "l_var",
            "cov": "l_cov",
        }
        self.record_every = int(record_every)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.history_path = os.path.join(self.metrics_dir, "history.jsonl")

    def _get_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        z = self.encoder(x, training=False)
        if self.compute_on == "projector" and self.projector is not None:
            z = self.projector(z, training=False)
        return z

    @staticmethod
    def _tf_avg_std(z: tf.Tensor) -> tf.Tensor:
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])
        std = tf.math.reduce_std(z2, axis=0)
        return tf.reduce_mean(std)

    @staticmethod
    def _tf_avg_offdiag_corr_sq(z: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])     # [N, D]
        n = tf.shape(z2)[0]
        d = tf.shape(z2)[1]
        mean = tf.reduce_mean(z2, axis=0, keepdims=True)
        zc = z2 - mean
        std = tf.math.reduce_std(zc, axis=0, keepdims=True)
        std = tf.where(std < eps, tf.ones_like(std), std)
        zn = zc / std
        corr = tf.matmul(zn, zn, transpose_a=True) / tf.cast(n, zn.dtype)
        eye = tf.eye(d, dtype=tf.bool)
        off = tf.boolean_mask(corr, ~eye)
        return tf.reduce_mean(tf.square(off))

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.record_every != 0:
            return

        rec = {"epoch": int(epoch + 1)}
        for pretty, key in self.loss_keys.items():
            if key in logs and logs[key] is not None:
                try:
                    rec[f"loss/{pretty}"] = float(logs[key])
                except Exception:
                    pass

        if self.sample_images is not None:
            x = self.sample_images
            if not tf.is_tensor(x):
                x = tf.convert_to_tensor(x)
            if x.dtype != tf.float32:
                x = tf.cast(x, tf.float32)
            z = self._get_embeddings(x)
            if z.dtype not in (tf.float32, tf.float64):
                z = tf.cast(z, tf.float32)
            try:
                rec["stats/avg_std"] = float(self._tf_avg_std(z).numpy().item())
                rec["stats/avg_offdiag_corr_sq"] = float(self._tf_avg_offdiag_corr_sq(z).numpy().item())
            except Exception:
                rec["stats/avg_std"] = float("nan")
                rec["stats/avg_offdiag_corr_sq"] = float("nan")

        with open(self.history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ============================== CLI / Device ===================================

def _preparse_device() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    a, _ = p.parse_known_args()
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return a.device


_DEVICE_FLAG = _preparse_device()


def decide_device(device_flag: str) -> str:
    if device_flag == "cpu":
        print("[resume] Forcing CPU mode per flag.")
        return "/CPU:0"
    print_devices()
    enable_memory_growth()
    if device_flag == "gpu":
        print("[resume] Requested GPU; will not fall back.")
        return "/GPU:0"
    ok = gpu_probe_ok()
    if not ok:
        print("[resume] GPU probe failed; pinning to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    p.add_argument("--ckpt", type=str, required=True, help="Path to .weights.h5 to load (full or encoder-only).")
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    p.add_argument("--proj-out", type=int, default=4096, help="Projector output width.")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector depth (1, 2, or 3).")
    p.add_argument("--lr", type=float, default=0.01, help="Base LR used for scheduling.")
    p.add_argument("--resume-lr", type=float, default=None, help="LR set immediately after resume.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay (if optimizer supports).")
    p.add_argument("--adaptive", action="store_true", help="Enable adaptive target schedules.")
    p.add_argument("--use-schedules", action="store_true", help="Enable cosine schedules.")
    p.add_argument("--initial-epoch", type=int, default=0, help="Epoch index to start from.")
    p.add_argument("--epochs", type=int, default=100, help="Final epoch count to train to.")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf/resumed_run", help="Where to write outputs.")
    # Safety knobs
    p.add_argument("--clipnorm", type=float, default=1.0, help="Gradient clip-norm for stability.")
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps post-resume.")
    p.add_argument("--bn-freeze-steps", type=int, default=0, help="Freeze BN updates for N steps.")
    p.add_argument("--loss-guard", type=float, default=1e12, help="Shrink LR if batch loss > threshold.")
    # Metrics knobs (match training)
    p.add_argument("--metrics-probe-batch", type=int, default=256, help="Probe batch for stats.")
    p.add_argument("--metrics-compute-on", choices=["projector", "encoder"], default="projector",
                   help="Where to compute embedding stats.")
    p.add_argument("--record-every", type=int, default=1, help="Record stats every N epochs.")
    return p.parse_args()


# ================================== main =======================================

def _take_single_view_batch(ds: tf.data.Dataset, size_limit: Optional[int]) -> Optional[tf.Tensor]:
    try:
        batch = next(iter(ds))
    except Exception:
        return None
    x = batch[0] if (isinstance(batch, (tuple, list)) and len(batch) >= 1) else batch
    if size_limit is not None:
        x = x[: int(size_limit)]
    return x


def main() -> None:
    """
    Resume training from existing weights with optional LR warmup and guards.
    """
    args = parse_args()
    set_mixed_precision(False)

    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(f"[resume] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
          f"initial_epoch={args.initial_epoch} final_epochs={args.epochs} steps/epoch={steps_per_epoch}")

    device_str = decide_device(_DEVICE_FLAG)
    print(f"[resume] Using device scope: {device_str}")

    run_dir = args.model_dir
    os.makedirs(run_dir, exist_ok=True)

    with tf.device(device_str):
        encoder = build_encoder(args.image_size)
        projector = build_projector(2048, args.proj_out, args.proj_layers)

        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
            adaptive=args.adaptive,
            use_schedules=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            bn_freeze_steps=args.bn_freeze_steps,
        )

        force_build_for_saving(trainer, encoder, projector, args.image_size)

        # Robust weight load (full trainer or partial-by-name)
        safe_load_trainer_weights(trainer, args.ckpt)

        # Optimizer preference: AdamW if available
        base_lr = args.lr if args.resume_lr is None else float(args.resume_lr)
        try:
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(
                learning_rate=base_lr,
                weight_decay=args.wd,
                clipnorm=(args.clipnorm if args.clipnorm > 0 else None),
            )
        except Exception:
            opt = keras.optimizers.Adam(
                learning_rate=base_lr,
                clipnorm=(args.clipnorm if args.clipnorm > 0 else None),
            )
        trainer.compile(optimizer=opt)

        # Align internal step counter for smooth schedules
        trainer.curr_step = int(args.initial_epoch) * int(steps_per_epoch)

        ckpt_path = os.path.join(run_dir, "vicreg_tf.weights.h5")

        sample_images = _take_single_view_batch(ds, size_limit=args.metrics_probe_batch)
        metrics_cb = VicRegMetricsLogger(
            run_dir=run_dir,
            encoder=encoder,
            projector=projector,
            sample_images=sample_images,
            compute_on=args.metrics_compute_on,
            loss_keys={"total": "loss", "align": "l_align", "var": "l_var", "cov": "l_cov"},
            record_every=args.record_every,
        )

        callbacks = [
            WarmupLR(base_lr=base_lr, warmup_steps=args.warmup_steps),
            LossExplosionGuard(threshold=float(args.loss_guard), factor=0.1),
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path, save_weights_only=True, monitor="loss", save_best_only=True, verbose=1),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(os.path.join(run_dir, "resume_log.csv"), append=True),
            metrics_cb,
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
