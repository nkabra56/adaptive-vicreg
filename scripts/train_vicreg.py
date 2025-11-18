"""
Script Title: VICReg / Adaptive VICReg Pretraining (TensorFlow + Keras)

What this script does (my words)
--------------------------------
I pretrain a backbone encoder + small projector with the VICReg objective.
The trainer (in `vicreg_tf.model.VICRegTrainer`) computes the three VICReg
terms (alignment, variance, covariance), applies optional *adaptive* targets
(gamma for variance floor, nu for correlation target), logs metrics for my
report, and periodically saves weights.

Why I wrote it this way
-----------------------
- I use a simple `tf.data` two-view pipeline so the trainer always receives
  paired augmented images (x1, x2) per batch.
- I keep the optimizer construction here and make learning-rate/weight-decay
  *cosine* schedules optional via a callback (pure Keras).
- I add a custom JSONL logger so `report_metrics.py` can produce plots/tables
  without peeking into TF Summary or other formats.
- **Adaptive VICReg** is a *feature flag*: passing `--adaptive` enables time-
  varying targets (gamma, nu). If I *omit* `--adaptive`, I run *exact baseline*
  VICReg (gamma=1.0, nu=0.0) with the same loss code—no branching elsewhere.

Artifacts (written under a timestamped run directory)
-----------------------------------------------------
• vicreg_full.weights.h5        -> weights for the whole trainer (encoder+projector)
• vicreg_encoder.weights.h5     -> encoder-only snapshot for downstream eval
• train_config.json             -> exact hyperparameters for reproducibility
• metrics/history.jsonl         -> one JSON object per recorded epoch

Example usage
-------------
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
  --run-name pretrain-c10_checktrainer \

Baseline VICReg (no adaptive behavior)
--------------------------------------
Just omit `--adaptive` and everything else is identical. This is important for
clean ablations (same code path, same losses; only targets differ).

Author: Nishant Kabra
Date: 11/18/2025
"""
from __future__ import annotations

# ── Make local `src/` importable so `from vicreg_tf import ...` resolves properly.
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]  # <repo>
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import datetime as _dt
import json
import os
from typing import Optional, List

import tensorflow as tf
from tensorflow import keras

# I re-export builders/utilities from the `vicreg_tf` package.
from vicreg_tf import (
    VICRegTrainer,
    VICRegWeights,
    build_dataset,
    build_encoder,
    build_projector,
    force_build_for_saving,
    gpu_probe_ok,
    print_devices,
    set_mixed_precision,
    steps_for_dataset,
    enable_memory_growth,
)
from vicreg_tf import schedules as sched  # cosine scaler (pure function)


# ===========================================================
# Cosine LR / WD schedules applied per *step* via a Callback.
# ===========================================================
class CosineScheduleCallback(keras.callbacks.Callback):
    """
    Per-step cosine scheduling for my optimizer using `vicreg_tf.schedules`.

    What I scale
    ------------
    • optimizer.learning_rate := base_lr * cosine_scaler(step / total_steps)
    • optimizer.weight_decay  := base_wd * cosine_scaler(step / total_steps)
      (only if the optimizer exposes `weight_decay`)

    Why I like this
    ---------------
    Step-based schedules are smoother than epoch-based jumps and independent of
    the dataloader's exact length.

    Parameters
    ----------
    optimizer : keras.optimizers.Optimizer
        The optimizer whose LR (and optional WD) I scale.
    total_steps : int
        Global number of steps = steps_per_epoch * epochs.
    base_lr : float
        Initial learning rate (scaled down over training).
    base_wd : float or None
        Initial weight decay (if not supported by the optimizer, this is ignored).
    verbose : int
        If >0, I print current LR (and WD) at each epoch start.
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
        # NOTE: I clamp step in case callbacks fire after total_steps due to bookkeeping.
        step = min(self._step, self.total_steps)

        # This call returns a Python float (eager-safe). I keep it simple:
        # the pure function takes step and total_steps and returns scale in [0,1].
        scale = float(sched.cosine_scaler(step=step, total_steps=self.total_steps))

        # 1) Scale LR every batch (TF/keras accepts assign() or attribute set depending on backend).
        lr = self.base_lr * scale
        try:
            self.opt.learning_rate.assign(lr)  # most opt objects have a TF variable
        except Exception:
            self.opt.learning_rate = lr        # graceful fallback if not a variable

        # 2) Scale WD if the optimizer supports it (AdamW, etc.).
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

        # Read back current LR (and WD if present) to print a nice header line.
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


# ===========================================================
# Minimal JSONL metrics logger for plotting/report generation
# ===========================================================
class VicRegMetricsLogger(keras.callbacks.Callback):
    """
    Log VICReg losses and simple embedding stats to JSONL for my plots.

    What I record
    -------------
    • loss/total, loss/align, loss/var, loss/cov  (if trainer exposes metrics)
    • stats/avg_std  (mean per-dim std across the current probe batch)
    • stats/avg_offdiag_corr_sq  (mean squared off-diagonal of corr matrix)

    Why this helps
    --------------
    These numbers feed directly into `report_metrics.py` to render training-
    curves and small sanity checks (e.g., variance collapse or redundancy).

    Parameters
    ----------
    run_dir : str
        Directory to write `<run_dir>/metrics/history.jsonl`.
    encoder : tf.keras.Model
        My backbone (used to compute probe features if `compute_on=encoder`).
    projector : tf.keras.Model or None
        My projector head (used if `compute_on=projector`).
    sample_images : tf.Tensor | None
        A small uint8/float32 single-view probe batch; if None, I only log loss terms.
    compute_on : {'projector','encoder'}
        Where I compute the probe stats.
    loss_keys : dict[str,str]
        Mapping from pretty names to keys present in Keras logs.
    record_every : int
        Frequency in epochs to write the record (to reduce overhead).
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
        }
        self.record_every = int(record_every)
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.history_path = os.path.join(self.metrics_dir, "history.jsonl")

    def _get_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        # Compute features from encoder (and projector if requested) *without* training-side effects.
        z = self.encoder(x, training=False)
        if self.compute_on == "projector" and self.projector is not None:
            z = self.projector(z, training=False)
        return z

    @staticmethod
    def _tf_avg_std(z: tf.Tensor) -> tf.Tensor:
        # Mean of per-dimension stds; a crude but useful collapse check.
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])  # [N, D]
        std = tf.math.reduce_std(z2, axis=0)      # [D]
        return tf.reduce_mean(std)                # scalar

    @staticmethod
    def _tf_avg_offdiag_corr_sq(z: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        # Compute corr(zn) = (zn^T zn) / N, then average squared off-diagonals.
        z2 = tf.reshape(z, [tf.shape(z)[0], -1])        # [N, D]
        n = tf.shape(z2)[0]
        d = tf.shape(z2)[1]
        mean = tf.reduce_mean(z2, axis=0, keepdims=True)
        zc = z2 - mean
        std = tf.math.reduce_std(zc, axis=0, keepdims=True)
        std = tf.where(std < eps, tf.ones_like(std), std)
        zn = zc / std                                    # standardized features
        corr = tf.matmul(zn, zn, transpose_a=True) / tf.cast(n, zn.dtype)  # [D, D]
        eye = tf.eye(d, dtype=tf.bool)
        off = tf.boolean_mask(corr, ~eye)
        return tf.reduce_mean(tf.square(off))

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.record_every != 0:
            return

        rec = {"epoch": int(epoch + 1)}

        # 1) Pull loss terms from Keras logs using the mapping I declared above.
        for pretty, key in self.loss_keys.items():
            if key in logs and logs[key] is not None:
                try:
                    rec[f"loss/{pretty}"] = float(logs[key])
                except Exception:
                    pass

        # 2) Optionally compute probe stats from a small held-out single-view batch.
        if self.sample_images is not None:
            x = self.sample_images
            if not tf.is_tensor(x):
                x = tf.convert_to_tensor(x)
            if x.dtype != tf.float32:
                x = tf.cast(x, tf.float32)

            z = self._get_embeddings(x)  # encoder->[projector]
            if z.dtype not in (tf.float32, tf.float64):
                z = tf.cast(z, tf.float32)
            try:
                rec["stats/avg_std"] = float(self._tf_avg_std(z).numpy().item())
                rec["stats/avg_offdiag_corr_sq"] = float(self._tf_avg_offdiag_corr_sq(z).numpy().item())
            except Exception:
                # If something odd happened (e.g., tracing), keep the row but set NaN.
                rec["stats/avg_std"] = float("nan")
                rec["stats/avg_offdiag_corr_sq"] = float("nan")

        with open(self.history_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


# ========================== Early device flag handling =========================
def _preparse_device_flag() -> str:
    # I parse --device before importing TF CUDA context to force CPU if requested.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    args, _ = p.parse_known_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide GPUs from TF
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    return args.device


_DEVICE_FLAG = _preparse_device_flag()


def decide_device(device_flag: str) -> str:
    """
    Choose a TF device string based on my preference and a quick GPU sanity probe.
    """
    if device_flag == "cpu":
        print("[train] Forcing CPU mode per flag.")
        return "/CPU:0"
    print_devices()          # print visible GPUs for debugging
    enable_memory_growth()   # prevent TF from greedily allocating all VRAM
    if device_flag == "gpu":
        print("[train] Requested GPU; will not fall back.")
        return "/GPU:0"
    ok = gpu_probe_ok()
    if not ok:
        print("[train] GPU probe failed; falling back to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


def parse_args() -> argparse.Namespace:
    """
    CLI flags I support for VICReg/Adaptive-VICReg pretraining.
    """
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
    # Adaptive & schedules (feature flags)
    p.add_argument("--adaptive", action="store_true", help="Enable adaptive gamma/nu targets. Omit for baseline VICReg.")
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
    return p.parse_args()


# ================================ main ========================================
def _take_single_view_batch(ds: tf.data.Dataset, size_limit: Optional[int]) -> Optional[tf.Tensor]:
    """
    Take one small single-view batch from a possibly two-view SSL dataset.
    Used only for probe statistics; this never influences training steps.
    """
    try:
        b = next(iter(ds))
    except Exception:
        return None
    x = b[0] if (isinstance(b, (tuple, list)) and len(b) >= 1) else b
    if size_limit is not None:
        x = x[: int(size_limit)]
    return x


def main() -> None:
    """
    Orchestrate data, models, optimizer, schedules, metrics, and saving.
    """
    args = parse_args()
    set_mixed_precision(False)  # I keep float32 for stability with VICReg

    # Dataset and steps/epoch (I keep two-view logic in `vicreg_tf.data`).
    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(
        f"[train] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
        f"epochs={args.epochs} steps/epoch={steps_per_epoch}"
    )

    # Output paths (either explicit path, or timestamped run folder).
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

    # Save the input config as JSON so I can reconstruct runs later.
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
                "use_schedules": args.use_schedules,
            },
            f,
            indent=2,
        )

    device_str = decide_device(_DEVICE_FLAG)
    print(f"[train] Using device scope: {device_str}")

    with tf.device(device_str):
        # -------------------------------
        # 1) Build encoder and projector.
        # -------------------------------
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)

        # -------------------------
        # 2) Construct the optimizer
        # -------------------------
        # I prefer AdamW; if unavailable (e.g., TF Addons mismatch), I fallback to Adam.
        try:
            import tensorflow_addons as tfa  # type: ignore
            opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            opt = keras.optimizers.Adam(learning_rate=args.lr)

        # ----------------------------------------------------------
        # 3) Build the trainer with my feature flags wired correctly.
        # ----------------------------------------------------------
        # NOTE: The *adaptive* flag only changes how gamma/nu are produced inside
        # the trainer. If `args.adaptive=False`, gamma=1.0 and nu=0.0 are used,
        # which matches *baseline VICReg* exactly.
        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.5),
            adaptive=args.adaptive,          # <── toggle: True => adaptive targets; False => baseline
            use_schedules=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            base_lr=args.lr,
            base_wd=args.wd,
        )
        trainer.compile(optimizer=opt)

        # --------------------------------------------------
        # 4) Force variable creation so save_weights works.
        # --------------------------------------------------
        force_build_for_saving(trainer, encoder, projector, args.image_size)

        # --------------------------------------------------
        # 5) Set up callbacks: checkpoint, NaN guard, logger
        # --------------------------------------------------
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

        # Internal metrics logger powering my plots (depends on small probe batch).
        sample_images = _take_single_view_batch(ds, size_limit=args.metrics_probe_batch)
        cbs.append(
            VicRegMetricsLogger(
                run_dir=out_dir,
                encoder=encoder,
                projector=projector,
                sample_images=sample_images,
                compute_on=args.metrics_compute_on,
                loss_keys={"total": "loss", "align": "l_align", "var": "l_var", "cov": "l_cov"},
                record_every=args.record_every,
            )
        )

        # Optional cosine LR/WD schedules (applied per *step*).
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

        # -----
        # 6) Go
        # -----
        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=cbs,
            verbose=1,
        )

        # Save encoder-only snapshot (downstream eval uses this).
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
