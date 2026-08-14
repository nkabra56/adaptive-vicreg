"""
VICReg / Adaptive VICReg pretraining (TensorFlow + Keras).

Pretrains a backbone encoder and small projector with the VICReg objective.
`vicreg_tf.model.VICRegTrainer` computes the three loss components and
applies two independent, optional mechanisms: `--adaptive` (per-step
loss-term weighting) and `--adaptive-targets` (a gamma/nu target schedule).
Omitting both flags runs exact baseline VICReg through the same code path.

Data is a `tf.data` two-view pipeline yielding paired augmented images
(x1, x2) per batch. LR/WD cosine schedules are optional, applied via a Keras
callback. A JSONL metrics logger records per-epoch losses and embedding
stats for `report_metrics.py`.

Artifacts, written under a timestamped run directory:
  vicreg_full.weights.h5     Full trainer weights (encoder + projector).
  vicreg_encoder.weights.h5  Encoder-only snapshot, for downstream eval.
  train_config.json          Hyperparameters, for reproducibility.
  metrics/history.jsonl      One JSON object per recorded epoch.

Example:
  python3 scripts/train_vicreg.py \
    --dataset cifar10 \
    --image-size 32 \
    --epochs 100 \
    --batch-size 512 \
    --feat-dim 2048 \
    --proj-out 4096 \
    --proj-layers 3 \
    --lr 0.01 \
    --wd 1e-6 \
    --adaptive \
    --use-schedules \
    --metrics-compute-on projector \
    --metrics-probe-batch 256 \
    --record-every 1 \
    --model-dir checkpoints_tf \
    --run-name pretrain-c10_checktrainer

Omit --adaptive (and --adaptive-targets) for baseline VICReg: same code
path, same losses, only the weights/targets differ.
"""
from __future__ import annotations

# Make local `src/` importable so `from vicreg_tf import ...` resolves.
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import datetime as _dt
import json
import os
from typing import Optional, List

import tensorflow as tf
from tensorflow import keras

from vicreg_tf import (
    VICRegTrainer,
    VICRegWeights,
    build_dataset,
    build_encoder,
    build_projector,
    force_build_for_saving,
    gpu_probe_ok,
    print_devices,
    set_global_seed,
    set_mixed_precision,
    steps_for_dataset,
    enable_memory_growth,
)
from vicreg_tf import schedules as sched


class CosineScheduleCallback(keras.callbacks.Callback):
    """
    Per-step cosine schedule for the optimizer's learning rate (and weight
    decay, if the optimizer exposes it), scaled by
    `cosine_scaler(step / total_steps)`. Step-based scheduling is smoother
    than epoch-based jumps and doesn't depend on the dataloader's exact
    length.

    Args:
        optimizer: The optimizer to scale.
        total_steps: steps_per_epoch * epochs.
        base_lr: Initial learning rate.
        base_wd: Initial weight decay, or None to leave weight decay alone.
        verbose: If >0, print the current LR (and WD) at each epoch start.
    """
    def __init__(
        self,
        optimizer: keras.optimizers.Optimizer,
        total_steps: int,
        base_lr: float,
        base_wd: Optional[float],
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.opt = optimizer
        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)
        self.base_wd = None if base_wd is None else float(base_wd)
        self.verbose = int(verbose)
        self._step = 0

        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.verbose:
            print(f"[schedules] total_steps={self.total_steps} base_lr={self.base_lr} base_wd={self.base_wd}")

    def on_train_batch_begin(self, batch: int, logs=None):
        # Clamp in case a callback fires after total_steps due to bookkeeping.
        step = min(self._step, self.total_steps)
        scale = float(sched.cosine_scaler(step=step, total_steps=self.total_steps))

        lr = self.base_lr * scale
        try:
            self.opt.learning_rate.assign(lr)
        except Exception:
            self.opt.learning_rate = lr

        if self.base_wd is not None and hasattr(self.opt, "weight_decay"):
            wd = self.base_wd * scale
            try:
                self.opt.weight_decay.assign(wd)
            except Exception:
                try:
                    self.opt.weight_decay = wd
                except Exception:
                    pass

        self._step += 1

    def on_epoch_begin(self, epoch: int, logs=None):
        if not self.verbose:
            return

        try:
            cur_lr = float(tf.keras.backend.get_value(self.opt.learning_rate))
        except Exception:
            cur_lr = float(self.opt.learning_rate)

        msg = f"[schedules] epoch {epoch+1:03d} | lr={cur_lr:.6f}"
        if self.base_wd is not None and hasattr(self.opt, "weight_decay"):
            try:
                cur_wd = float(tf.keras.backend.get_value(self.opt.weight_decay))
            except Exception:
                cur_wd = float(self.opt.weight_decay)
            msg += f" wd={cur_wd:.6f}"
        print(msg)


class VicRegMetricsLogger(keras.callbacks.Callback):
    """
    Append one JSON record per recorded epoch to
    `<run_dir>/metrics/history.jsonl`, for `report_metrics.py` to consume.

    Records loss/total, loss/align, loss/var, loss/cov (and the realized
    loss weights, if present in Keras logs), plus stats/avg_std and
    stats/avg_offdiag_corr_sq computed from a fixed probe batch.

    Args:
        run_dir: Directory to write metrics/history.jsonl under.
        encoder: Backbone, used to compute probe features.
        projector: Projector head, used if `compute_on="projector"`.
        sample_images: Fixed single-view probe batch; if None, only loss
            terms are logged.
        compute_on: Where to compute embedding stats, "projector" or "encoder".
        loss_keys: Maps output record keys to keys present in Keras logs.
        record_every: Write a record only every N epochs.
    """
    def __init__(
        self,
        run_dir: str,
        encoder: tf.keras.Model,
        projector: Optional[tf.keras.Model],
        sample_images: Optional[tf.Tensor],
        compute_on: str = "projector",
        loss_keys: Optional[dict] = None,
        record_every: int = 1,
    ) -> None:
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
            "w_sim": "w_sim",
            "w_var": "w_var",
            "w_cov": "w_cov",
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
        """Mean of per-dimension std across the batch; a crude collapse check."""
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])
        std = tf.math.reduce_std(z2, axis=0)
        return tf.reduce_mean(std)

    @staticmethod
    def _tf_avg_offdiag_corr_sq(z: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        """Mean squared off-diagonal entry of the standardized correlation matrix."""
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])
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
                # Tracing or another transient issue; keep the row but mark it NaN.
                rec["stats/avg_std"] = float("nan")
                rec["stats/avg_offdiag_corr_sq"] = float("nan")

        with open(self.history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _preparse_device_flag() -> str:
    """Parse --device before TF initializes CUDA, so CPU-only mode can hide GPUs in time."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args, _ = p.parse_known_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return args.device


_DEVICE_FLAG = _preparse_device_flag()


def decide_device(device_flag: str) -> str:
    """Resolve --device to a TF device string, probing the GPU in "auto" mode."""
    if device_flag == "cpu":
        print("[train] Forcing CPU mode per flag.")
        return "/CPU:0"
    print_devices()
    enable_memory_growth()
    if device_flag == "gpu":
        print("[train] Requested GPU; will not fall back.")
        return "/GPU:0"
    ok = gpu_probe_ok()
    if not ok:
        print("[train] GPU probe failed; falling back to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    # Data & schedule
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    # Model
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width.")
    p.add_argument("--proj-out", type=int, default=4096, help="Projector output width.")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector depth (1,2,3).")
    # Optimizer
    p.add_argument("--lr", type=float, default=0.01, help="Base learning rate.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay if optimizer supports it.")
    # Adaptive mechanisms
    p.add_argument("--adaptive", action="store_true",
                   help="Enable adaptive loss-term weighting (AdaptiveReweighter: EMA magnitude "
                        "balancing + embedding-health reactive boost). This is 'Adaptive VICReg'. "
                        "Omit for baseline VICReg (constant weights).")
    p.add_argument("--adaptive-targets", action="store_true",
                   help="Separate, optional mechanism: ramp gamma/nu targets over training "
                        "instead of the baseline constants (gamma=1.0, nu=0.0). Can be combined "
                        "with --adaptive.")
    p.add_argument("--ema-decay", type=float, default=0.98,
                   help="EMA decay for AdaptiveReweighter's loss-magnitude and embedding-health "
                        "tracking (only used with --adaptive).")
    p.add_argument("--var-boost-k", type=float, default=2.0,
                   help="Reactive-boost strength for the variance weight when avg std drifts "
                        "below target (only used with --adaptive).")
    p.add_argument("--cov-boost-k", type=float, default=2.0,
                   help="Reactive-boost strength for the covariance weight when avg off-diagonal "
                        "correlation drifts above target (only used with --adaptive).")
    p.add_argument("--use-schedules", action="store_true", help="Apply cosine LR/WD schedules per step.")
    # Output / run naming
    p.add_argument("--model-dir", type=str, default="checkpoints_tf", help="Root output dir.")
    p.add_argument("--run-name", type=str, default=None, help="Subfolder prefix; timestamp is appended.")
    p.add_argument("--ckpt-out", type=str, default=None,
                   help="Optional explicit full weights path (overrides model-dir).")
    # Metrics logger knobs
    p.add_argument("--metrics-probe-batch", type=int, default=256,
                   help="Size of the fixed single-view probe batch for stats.")
    p.add_argument("--metrics-compute-on", choices=["projector", "encoder"], default="projector",
                   help="Where to compute embedding stats.")
    p.add_argument("--record-every", type=int, default=1,
                   help="Record internal metrics every N epochs.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed Python/NumPy/TF for reproducibility. Omit for nondeterministic runs.")
    return p.parse_args()


def _take_single_view_batch(ds: tf.data.Dataset, size_limit: Optional[int]) -> Optional[tf.Tensor]:
    """Take one small single-view batch from the two-view dataset, for probe statistics only (never used for training)."""
    try:
        b = next(iter(ds))
    except Exception:
        return None
    x = b[0] if (isinstance(b, (tuple, list)) and len(b) >= 1) else b
    if size_limit is not None:
        x = x[: int(size_limit)]
    return x


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

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
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
                "adaptive": args.adaptive,
                "adaptive_targets": args.adaptive_targets,
                "ema_decay": args.ema_decay,
                "var_boost_k": args.var_boost_k,
                "cov_boost_k": args.cov_boost_k,
                "use_schedules": args.use_schedules,
                "seed": args.seed,
            },
            f,
            indent=2,
        )

    device_str = decide_device(_DEVICE_FLAG)
    print(f"[train] Using device scope: {device_str}")

    with tf.device(device_str):
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)

        # Prefer native Keras AdamW, then TFA's AdamW, then plain Adam.
        try:
            opt = keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            try:
                import tensorflow_addons as tfa  # type: ignore
                opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
            except Exception:
                opt = keras.optimizers.Adam(learning_rate=args.lr)

        # `adaptive` toggles per-step loss-term weighting (AdaptiveReweighter);
        # `adaptive_targets` independently toggles the gamma/nu schedule. With
        # both off, this is exactly baseline VICReg.
        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
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

        cbs: List[keras.callbacks.Callback] = [
            keras.callbacks.ModelCheckpoint(
                filepath=full_ckpt,
                save_weights_only=True,
                monitor="loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
        ]

        sample_images = _take_single_view_batch(ds, size_limit=args.metrics_probe_batch)
        cbs.append(
            VicRegMetricsLogger(
                run_dir=out_dir,
                encoder=encoder,
                projector=projector,
                sample_images=sample_images,
                compute_on=args.metrics_compute_on,
                loss_keys={
                    "total": "loss", "align": "l_align", "var": "l_var", "cov": "l_cov",
                    "w_sim": "w_sim", "w_var": "w_var", "w_cov": "w_cov",
                },
                record_every=args.record_every,
            )
        )

        if args.use_schedules:
            total_steps = steps_per_epoch * args.epochs
            cbs.insert(
                0,
                CosineScheduleCallback(
                    optimizer=opt,
                    total_steps=total_steps,
                    base_lr=args.lr,
                    base_wd=args.wd,
                    verbose=1,
                ),
            )

        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=cbs,
            verbose=1,
        )

        encoder.save_weights(enc_ckpt)

    print(
        f"[train] Done.\n"
        f"  Encoder -> {enc_ckpt}\n"
        f"  Full    -> {full_ckpt}\n"
        f"  Config  -> {os.path.join(out_dir, 'train_config.json')}\n"
        f"  Metrics -> {os.path.join(out_dir, 'metrics', 'history.jsonl')}"
    )


if __name__ == "__main__":
    main()
