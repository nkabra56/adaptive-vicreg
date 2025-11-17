"""
Script Title: Resume VICReg Pretraining From Weights

What this script does
---------------------
Rebuilds the encoder and projector, loads an existing checkpoint (full trainer
weights or encoder-only weights), and resumes training safely. Includes:
  • Optional LR warmup after resume
  • Optional BatchNorm update freeze for the first N steps
  • Loss explosion guard that shrinks LR if a batch loss spikes
  • The SAME per-epoch metrics logging as train_vicreg.py:
      - loss/total, loss/align, loss/var, loss/cov  (if exposed by trainer)
      - stats/avg_std
      - stats/avg_offdiag_corr_sq
    saved to <model_dir>/metrics/history.jsonl

Typical usage
-------------
python3 scripts/resume_pretrain.py \
  --ckpt checkpoints_tf/pretrain-.../vicreg_full.weights.h5 \
  --dataset cifar10 \
  --image-size 32 \
  --batch-size 256 \
  --proj-out 8192 \
  --proj-layers 3 \
  --epochs 200 \
  --initial-epoch 80 \
  --lr 0.2 \
  --resume-lr 0.1 \
  --wd 1e-6 \
  --warmup-steps 500 \
  --bn-freeze-steps 200 \
  --adaptive \
  --use-schedules \
  --record-every 1 \
  --metrics-probe-batch 256 \
  --metrics-compute-on projector \
  --device auto

Notes
-----
• You can resume from an encoder-only checkpoint; the projector starts fresh.
• This script writes `vicreg_tf.weights.h5` (best-so-far), a CSV log, and
  metrics JSONL under <model_dir>/metrics/history.jsonl (same keys as train).
• If your trainer exposes `l_align`, `l_var`, `l_cov` via add_metric(...),
  those will be recorded automatically.

Author: Nishant Kabra
Date: 11/16/2025
"""
from __future__ import annotations

# Ensure local package "vicreg_tf" under <repo>/src is importable ---
# Resolves imports when running like: python3 scripts/resume_pretrain.py
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]   # <repo>
_SRC_DIR = _REPO_ROOT / "src"                      # <repo>/src

# Verify expected layout and put src at the *front* of sys.path so local code wins
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# -------------------------------------------------------------------------------

import argparse
import os
import json
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

# -------------------- Metrics logger (identical behavior to train_vicreg.py) ---
class VicRegMetricsLogger(keras.callbacks.Callback):
    """
    Logs VICReg / Adaptive-VICReg internal metrics during training.

    Purpose
    -------
    On each epoch end (optionally every N epochs), this callback:
      • reads loss components from Keras logs (if trainer exposes them),
      • computes embedding statistics on a fixed probe batch:
           - stats/avg_std: average standard deviation across embedding dims
           - stats/avg_offdiag_corr_sq: mean squared off-diagonal correlation
      • appends a JSON object to `<run_dir>/metrics/history.jsonl`.

    Parameters
    ----------
    run_dir : str
        Folder used as the run directory (here: `args.model_dir`).
        We create `<run_dir>/metrics/history.jsonl` to store per-epoch rows.
    encoder : tf.keras.Model
        Backbone encoder. Used to compute features on the probe batch.
    projector : tf.keras.Model or None
        Projection head. If `compute_on='projector'`, embeddings are
        `projector(encoder(x))`; else they are `encoder(x)`.
    sample_images : tf.Tensor
        A small single-view batch `[N, H, W, C]` used only for metrics.
    compute_on : {'encoder','projector'}
        Where to compute the stats. 'projector' is recommended for variance/cov.
    loss_keys : dict[str, str]
        Mapping from pretty names to keys in `logs`.
        Defaults: {'total':'loss','align':'l_align','var':'l_var','cov':'l_cov'}.
    record_every : int
        Record every N epochs. Use >1 to reduce overhead if needed.
    """

    def __init__(
        self,
        run_dir: str,
        encoder: tf.keras.Model,
        projector: tf.keras.Model | None,
        sample_images: tf.Tensor | None,
        compute_on: str = "projector",
        loss_keys: dict[str, str] | None = None,
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

        # Ensure metrics directory exists and pick the history file path
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.history_path = os.path.join(self.metrics_dir, "history.jsonl")

    def _get_embeddings(self, x: tf.Tensor) -> tf.Tensor:
        """Compute embeddings on the chosen stage without gradient tracking."""
        z = self.encoder(x, training=False)  # encoder forward pass
        if self.compute_on == "projector" and self.projector is not None:
            z = self.projector(z, training=False)  # projector forward pass
        return z

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Collect metrics at epoch end and append to the JSONL file."""
        logs = logs or {}

        # Respect record_every to reduce overhead if requested
        if (epoch + 1) % self.record_every != 0:
            return

        record: dict[str, float | int] = {"epoch": int(epoch + 1)}

        # 1) Copy loss components from Keras logs if present
        for pretty, key in self.loss_keys.items():
            if key in logs and logs[key] is not None:
                try:
                    record[f"loss/{pretty}"] = float(logs[key])
                except Exception:
                    pass

        # ---------------- local helpers ----------------
        def _to_numpy(a):
            """Best-effort conversion to a NumPy array without breaking if TF is absent."""
            import numpy as np
            try:
                import tensorflow as tf  # noqa: F401  (runtime import)
                if tf.is_tensor(a):
                    return a.numpy()
            except Exception:
                pass

            if hasattr(a, "np"):
                try:
                    return np.array(a.np())
                except Exception:
                    pass

            return np.array(a)

        def _avg_std_impl(z_np) -> float:
            """Average per-dimension std across the batch."""
            import numpy as np
            z2 = z_np
            if z2.ndim > 2:
                # Flatten all non-batch dims into one embedding dim
                z2 = z2.reshape(z2.shape[0], -1)
            if z2.size == 0 or z2.shape[0] == 0:
                return float("nan")
            std = z2.std(axis=0, ddof=0)
            return float(np.mean(std))

        def _avg_offdiag_corr_sq_impl(z_np) -> float:
            """Mean squared off-diagonal correlation between embedding dims."""
            import numpy as np
            z2 = z_np
            if z2.ndim > 2:
                z2 = z2.reshape(z2.shape[0], -1)
            n, d = z2.shape
            if d <= 1 or n == 0:
                return 0.0

            # Normalize each dim to zero-mean, unit-variance (guard small-variance dims)
            zc = z2 - z2.mean(axis=0, keepdims=True)
            std = zc.std(axis=0, ddof=0, keepdims=True)
            std = np.where(std < 1e-12, 1.0, std)
            zn = zc / std

            # Correlation ≈ (zn^T zn) / n  (since zn is standardized per dim)
            corr = (zn.T @ zn) / float(n)

            # Off-diagonal mean of squared entries
            mask = ~np.eye(d, dtype=bool)
            off = corr[mask]
            return float(np.mean(off * off))

        # 2) Embedding-level statistics on the fixed probe batch
        if getattr(self, "sample_images", None) is not None:
            x = self.sample_images
            z = self._get_embeddings(x)
            z_np = _to_numpy(z)

            try:
                record["stats/avg_std"] = _avg_std_impl(z_np)
            except Exception:
                record["stats/avg_std"] = float("nan")

            try:
                record["stats/avg_offdiag_corr_sq"] = _avg_offdiag_corr_sq_impl(z_np)
            except Exception:
                record["stats/avg_offdiag_corr_sq"] = float("nan")

        # 3) Append one JSON object per line to the history file
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# --------------------------- Helpers shared with train_vicreg -------------------
def _take_single_view_batch(ds: tf.data.Dataset, size_limit: int | None) -> tf.Tensor | None:
    """
    Take one small, single-view batch from a possibly two-view SSL dataset.

    Parameters
    ----------
    ds : tf.data.Dataset
        The same dataset you pass to `fit()`. Often yields `(view1, view2)`.
    size_limit : int or None
        Optionally slice the batch to this many samples to keep metrics cheap.

    Returns
    -------
    tf.Tensor or None
        A tensor `[N, H, W, C]` if available; otherwise None (metrics will skip).
    """
    try:
        batch = next(iter(ds))
    except Exception:
        return None

    # If dataset yields (view1, view2), we take the first view for probing.
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        x = batch[0]
    else:
        x = batch

    if size_limit is not None:
        x = x[: int(size_limit)]
    return x


# Parse the device early for CUDA visibility decisions.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def parse_args() -> argparse.Namespace:
    """
    Parse CLI options for resuming a VICReg run.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--ckpt", type=str, required=True, help="Path to .weights.h5 checkpoint to load.")
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    p.add_argument("--proj-out", type=int, default=8192, help="Projector output width.")
    p.add_argument("--proj-layers", type=int, default=3, help="Projector depth (1, 2, or 3).")
    p.add_argument("--lr", type=float, default=0.2, help="Base LR used for scheduling.")
    p.add_argument("--resume-lr", type=float, default=None, help="LR set immediately after resume.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay (if optimizer supports).")
    p.add_argument("--adaptive", action="store_true", help="Enable adaptive target schedules.")
    p.add_argument("--use-schedules", action="store_true", help="Enable cosine schedules.")
    p.add_argument("--initial-epoch", type=int, default=0, help="Epoch index to start from.")
    p.add_argument("--epochs", type=int, default=100, help="Total epochs to train to.")
    p.add_argument("--model-dir", type=str, default="checkpoints_tf", help="Where to write outputs.")
    # Safety knobs:
    p.add_argument("--clipnorm", type=float, default=1.0, help="Gradient clip-norm for stability.")
    p.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps post-resume.")
    p.add_argument("--bn-freeze-steps", type=int, default=0, help="Freeze BN updates for N steps.")
    p.add_argument("--loss-guard", type=float, default=1e12, help="Shrink LR if batch loss > threshold.")
    # Metrics logger knobs (match train_vicreg.py)
    p.add_argument("--metrics-probe-batch", type=int, default=256,
                   help="Size of the probe batch for internal metrics (single view).")
    p.add_argument("--metrics-compute-on", choices=["projector", "encoder"], default="projector",
                   help="Where to compute embedding stats.")
    p.add_argument("--record-every", type=int, default=1,
                   help="Record internal metrics every N epochs (to reduce overhead).")
    return p.parse_args()


class WarmupLR(keras.callbacks.Callback):
    """
    Linear learning-rate warmup for the first N steps after resuming.

    Parameters
    ----------
    base_lr : float
        Target learning rate after warmup.
    warmup_steps : int
        Number of optimizer steps over which to increase LR from 10% to 100%.

    Notes
    -----
    Uses the trainer's `curr_step` counter to compute progress during warmup.
    """

    def __init__(self, base_lr: float, warmup_steps: int):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(0, warmup_steps))

    def on_train_batch_begin(self, batch, logs=None):
        """
        Scale LR linearly at the start of each training batch during warmup.

        The LR schedule during warmup is:
            lr_t = 0.1 * base + 0.9 * base * (step / warmup_steps)

        After warmup_steps are consumed, this callback does nothing.
        """
        if self.warmup_steps <= 0:
            return
        step = int(self.model.curr_step)  # number of steps completed so far
        if step < self.warmup_steps:
            scale = 0.1 + 0.9 * (step + 1) / float(self.warmup_steps)
            new_lr = self.base_lr * scale
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)


class LossExplosionGuard(keras.callbacks.Callback):
    """
    Shrink the learning rate if a single batch reports an absurdly large loss.

    Parameters
    ----------
    threshold : float
        Loss value beyond which we consider the batch numerically unstable.
    factor : float
        Multiplicative factor to shrink the LR (e.g., 0.1 -> 10x reduction).
    """

    def __init__(self, threshold: float = 1e8, factor: float = 0.1):
        super().__init__()
        self.threshold = float(threshold)
        self.factor = float(factor)

    def on_train_batch_end(self, batch, logs=None):
        """
        Check the reported loss after each batch and reduce LR if it is too large.

        This helps recover from occasional spikes due to bad augment combinations
        or numerical corner cases.
        """
        if not logs:
            return
        loss = float(logs.get("loss", 0.0))
        if loss > self.threshold:
            lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = max(lr * self.factor, 1e-6)  # keep a sane lower bound
            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            print(
                f"[loss-guard] loss={loss:.3e} > {self.threshold:.1e} -> "
                f"lr {lr:.2e} -> {new_lr:.2e}"
            )


def decide_device(device_flag: str) -> str:
    """
    Resolve device placement given user preference and a quick GPU probe.

    Parameters
    ----------
    device_flag : str
        "auto", "gpu", or "cpu".

    Returns
    -------
    str
        TensorFlow device string.
    """
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


def main() -> None:
    """
    Entry point for resuming training.

    Steps
    -----
    1) Build data pipeline and compute steps per epoch.
    2) Build encoder, projector, and VICRegTrainer; create variables for saving.
    3) Load checkpoint robustly (structural match first; fallback to by_name).
    4) Compile with requested optimizer; set `curr_step` to reflect progress.
    5) Train with callbacks for warmup, loss guard, best-weights checkpoint,
       and metrics logging identical to train_vicreg.py.
    """
    args = parse_args()
    set_mixed_precision(False)

    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(
        f"[resume] dataset={args.dataset} img={args.image_size} "
        f"bs={args.batch_size} initial_epoch={args.initial_epoch} "
        f"final_epochs={args.epochs} steps/epoch={steps_per_epoch}"
    )

    device_str = decide_device(_pre_args.device)
    print(f"[resume] Using device scope: {device_str}")

    # Use <model_dir> as the "run directory" so metrics go to:
    #   <model_dir>/metrics/history.jsonl
    run_dir = args.model_dir
    os.makedirs(run_dir, exist_ok=True)

    with tf.device(device_str):
        # Build models. The encoder's final `feat` has width 2048 by default.
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

        # Make sure variables exist for save/load on a subclassed model.
        force_build_for_saving(trainer, encoder, projector, args.image_size)

        # Load the checkpoint robustly. If loading the full trainer fails due to
        # shape/name mismatches, try a by_name partial restore.
        safe_load_trainer_weights(trainer, args.ckpt)

        # Choose optimizer and set the "resume" learning rate if provided.
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

        # Align the trainer's internal step counter so schedules resume smoothly.
        trainer.curr_step = int(args.initial_epoch) * int(steps_per_epoch)

        ckpt_path = os.path.join(run_dir, "vicreg_tf.weights.h5")

        # --- Metrics logger setup (identical behavior/keys as train_vicreg.py) ---
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
                filepath=ckpt_path,
                save_weights_only=True,
                monitor="loss",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(os.path.join(run_dir, "resume_log.csv"), append=True),
            metrics_cb,  # <— add metrics logger
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
