"""
Script Title: VICReg / Adaptive VICReg Pretraining (TensorFlow + Keras)

What this script does
---------------------
Builds an encoder and a projector, wraps them in a subclassed Keras Model
(`VICRegTrainer`) that implements the VICReg loss (invariance/variance/
covariance), and trains with `model.fit`. The script writes:
  • `vicreg_full.weights.h5`    (trainer weights: encoder + projector)
  • `vicreg_encoder.weights.h5` (encoder-only weights for downstream eval)
  • `train_config.json`         (training configuration snapshot)
  • `metrics/history.jsonl`     (per-epoch JSONL records for plots/tables)

It also attaches a metrics-logging callback that:
  • computes average embedding standard deviation (variance control),
  • computes mean squared off-diagonal correlation (redundancy control),
  • picks up loss component scalars (if your trainer exposes them as Keras metrics),
  • appends everything to `<run_dir>/metrics/history.jsonl`.

Typical usage
-------------
python3 scripts/train_vicreg.py \
  --dataset cifar10 \
  --image-size 32 \
  --epochs 100 \
  --batch-size 256 \
  --feat-dim 2048 \
  --proj-out 8192 \
  --proj-layers 3 \
  --lr 0.1 \
  --wd 1e-6 \
  --adaptive \
  --use-schedules \
  --model-dir checkpoints_tf \
  --run-name pretrain-c10_model6_e100 \
  --device auto

Notes
-----
• Use `--device cpu` if CUDA kernels are unavailable; `auto` will probe GPU
  once and fall back to CPU if needed.
• We save **weights only**. The subclassed trainer should implement `get_config()`
  to silence Keras’ non-serializable args warning. If you still see the warning,
  add a minimal `get_config()` to your trainer and projector (see `vicreg_tf.model`).
• The encoder exposes a named `feat` tensor; downstream eval scripts load
  `vicreg_encoder.weights.h5` and use that `feat` vector.
• Metrics logger records the internal stats for your plots in `report_metrics.py`.
  If your trainer doesn’t add `l_align`, `l_var`, `l_cov` to logs, the callback
  still records embedding stats; add:
      self.add_metric(l_align, name="l_align", aggregation="mean")
      self.add_metric(l_var,   name="l_var",   aggregation="mean")
      self.add_metric(l_cov,   name="l_cov",   aggregation="mean")
  inside your trainer’s `train_step` to get the loss curves too.

Author: Nishant Kabra
Date: 11/16/2025
"""
from __future__ import annotations

# --- Make sure local package "vicreg_tf" (under <repo>/src) is importable. -----
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]   # <repo>
_SRC_DIR = _REPO_ROOT / "src"                      # <repo>/src

# Prepend to sys.path so your local code wins over any installed modules.
if not _SRC_DIR.exists():
    raise RuntimeError(f"Could not find expected source directory: {_SRC_DIR}")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# ------------------------------------------------------------------------------

import argparse
import datetime as _dt
import json
import os
import typing as _t
import numpy as np

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
    set_mixed_precision,
    steps_for_dataset,
    enable_memory_growth,
)

# ------------------------------------------------------------------------------
# Metrics helpers imported from my vicreg_tf.report_metrics module.
# If I move the functions, I will update these imports accordingly.
# ------------------------------------------------------------------------------
from vicreg_tf import report_metrics as rm

# ------------------------------------------------------------------------------
# Cosine schedules are taken from my vicreg_tf.schedules module.
# I apply them per step using a lightweight callback below.
# ------------------------------------------------------------------------------
from vicreg_tf import schedules as sched


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
        The run folder created by this script (contains weights + config).
        I create `<run_dir>/metrics/history.jsonl` to store per-epoch rows.
    encoder : tf.keras.Model
        My backbone encoder. Used to compute features on the probe batch.
    projector : tf.keras.Model or None
        My projection head. If `compute_on='projector'`, embeddings are
        computed as `projector(encoder(x))`; else they are `encoder(x)`.
    sample_images : tf.Tensor
        A small single-view batch `[N, H, W, C]` used only for metrics.
        I keep N modest (e.g., 256) so this callback is cheap.
    compute_on : {'encoder','projector'}
        Where to compute the stats. 'projector' is recommended for variance/cov.
    loss_keys : dict[str, str]
        Mapping from pretty names to keys in `logs` (Keras on_epoch_end logs).
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
        """Collect metrics at epoch end and append to the JSONL file.

        I use helpers in `vicreg_tf.report_metrics`:
          - stats/avg_std:              rm.compute_embedding_avg_std(z)
          - stats/avg_offdiag_corr_sq:  rm.compute_avg_offdiag_corr_sq(z)
        """
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

        # 2) Embedding-level statistics on the fixed probe batch
        if getattr(self, "sample_images", None) is not None:
            x = self.sample_images
            z = self._get_embeddings(x)

            try:
                avg_std_raw = rm.compute_embedding_avg_std(z)
                avg_std = float(avg_std_raw.numpy() if hasattr(avg_std_raw, "numpy") else float(avg_std_raw))
                record["stats/avg_std"] = avg_std
            except Exception:
                record["stats/avg_std"] = float("nan")

            try:
                avg_corr_sq_raw = rm.compute_avg_offdiag_corr_sq(z)
                avg_corr_sq = float(avg_corr_sq_raw.numpy() if hasattr(avg_corr_sq_raw, "numpy") else float(avg_corr_sq))
                record["stats/avg_offdiag_corr_sq"] = avg_corr_sq
            except Exception:
                record["stats/avg_offdiag_corr_sq"] = float("nan")

        # 3) Append one JSON object per line to the history file
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")


class CosineScheduleCallback(keras.callbacks.Callback):
    """
    Per-step cosine scheduling for my optimizer using `vicreg_tf.schedules`.

    What I scale
    ------------
    • optimizer.learning_rate  := base_lr * cosine_scaler(step, total_steps)
    • optimizer.weight_decay   := base_wd * cosine_scaler(step, total_steps)  (if optimizer exposes it)

    How I count steps
    -----------------
    I count optimizer steps inside `on_train_batch_begin`. Total steps is
    `steps_per_epoch * epochs`, so the schedule spans the entire run.

    Notes
    -----
    • This uses the same math as my `vicreg_tf.schedules.cosine_scaler`.
    • If my optimizer does not have a `weight_decay` attribute (e.g., plain Adam),
      only the learning rate is scaled.
    """

    def __init__(
        self,
        optimizer: keras.optimizers.Optimizer,
        total_steps: int,
        base_lr: float,
        base_wd: float | None,
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
            raise ValueError("total_steps must be a positive integer")
        if self.verbose:
            print(f"[schedules] total_steps={self.total_steps} base_lr={self.base_lr} base_wd={self.base_wd}")

    def on_train_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        # Convert to a clamped global step in [0, total_steps]
        step = min(self._step, self.total_steps)
        scale = float(sched.cosine_scaler(step=step, total_steps=self.total_steps))

        # Scale learning rate (prefer assign if it's a tf.Variable)
        lr = self.base_lr * scale
        try:
            self.opt.learning_rate.assign(lr)
        except Exception:
            self.opt.learning_rate = lr

        # Scale weight decay if the optimizer exposes it
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

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
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


# ------------------------------------------------------------------------------
# Early parse for device flag (so I can set CUDA visibility prior to TF init).
# ------------------------------------------------------------------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument(
    "--device",
    choices=["auto", "gpu", "cpu"],
    default="auto",
    help="Device placement: try GPU, force GPU, or force CPU.",
)
_pre_args, _ = _pre.parse_known_args()

# Hide GPUs if I explicitly asked for CPU (must be set before TF loads).
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Reduce TF logging noise unless there is an error.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def parse_args() -> argparse.Namespace:
    """
    Parse all CLI arguments for pretraining.

    Returns
    -------
    argparse.Namespace
        Holds all CLI options as attributes.
    """
    p = argparse.ArgumentParser(parents=[_pre])
    # --- Data & schedule ------------------------------------------------------
    p.add_argument("--dataset", type=str, default="cifar10", help="cifar10 or cifar100.")
    p.add_argument("--image-size", type=int, default=32, help="Square crop size.")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    p.add_argument("--batch-size", type=int, default=256, help="Global batch size.")
    # --- Model widths ---------------------------------------------------------
    p.add_argument("--feat-dim", type=int, default=2048, help="Encoder feature width.")
    p.add_argument("--proj-out", type=int, default=8192, help="Projector output width.")
    p.add_argument(
        "--proj-layers",
        type=int,
        default=3,
        help="Number of MLP layers in the projector (1, 2, or 3).",
    )
    # --- Optimizer ------------------------------------------------------------
    p.add_argument("--lr", type=float, default=0.2, help="Base learning rate.")
    p.add_argument("--wd", type=float, default=1e-6, help="Weight decay (if supported).")
    # --- Adaptive & schedules -------------------------------------------------
    p.add_argument("--adaptive", action="store_true", help="Enable adaptive gamma/nu targets.")
    p.add_argument("--use-schedules", action="store_true", help="Apply cosine schedules.")
    # --- Output layout --------------------------------------------------------
    p.add_argument("--model-dir", type=str, default="checkpoints_tf", help="Root output dir.")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional subfolder prefix; if omitted, a timestamp is used.",
    )
    p.add_argument(
        "--ckpt-out",
        type=str,
        default=None,
        help="Optional explicit path for full weights (overrides model-dir).",
    )
    # --- Metrics logger knobs -------------------------------------------------
    p.add_argument(
        "--metrics-probe-batch",
        type=int,
        default=256,
        help="Size of the probe batch for internal metrics (single view).",
    )
    p.add_argument(
        "--metrics-compute-on",
        choices=["projector", "encoder"],
        default="projector",
        help="Where to compute embedding stats.",
    )
    p.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record internal metrics every N epochs (to reduce overhead).",
    )
    return p.parse_args()


def decide_device(device_flag: str) -> str:
    """
    Choose a TF device string based on my preference and a quick GPU sanity probe.

    Parameters
    ----------
    device_flag : str
        One of {"auto","gpu","cpu"} from the CLI.

    Returns
    -------
    str
        A TensorFlow device string, e.g. "/GPU:0" or "/CPU:0".
    """
    if device_flag == "cpu":
        print("[train] Forcing CPU mode per flag.")
        return "/CPU:0"

    # Print visible devices and enable memory growth to avoid pre-allocating all VRAM.
    print_devices()
    enable_memory_growth()

    if device_flag == "gpu":
        print("[train] Requested GPU; will not fall back.")
        return "/GPU:0"

    # device_flag == "auto": try fast probe; fall back to CPU if kernels unavailable.
    ok = gpu_probe_ok()
    if not ok:
        print("[train] GPU probe failed; falling back to CPU.")
    return "/GPU:0" if ok else "/CPU:0"


def _take_single_view_batch(
    ds: tf.data.Dataset, size_limit: int | None
) -> tf.Tensor | None:
    """
    Take one small, single-view batch from a possibly two-view SSL dataset.

    Parameters
    ----------
    ds : tf.data.Dataset
        The same dataset I pass to `fit()`. Often yields `(view1, view2)`.
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

    # If dataset yields (view1, view2), I take the first view for probing.
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        x = batch[0]
    else:
        x = batch

    # Optionally slice to keep metrics lightweight.
    if size_limit is not None:
        x = x[: int(size_limit)]

    return x


def main() -> None:
    """
    Entry point: builds models, prepares data, runs training, and writes checkpoints.

    This function orchestrates:
      1) argparse + recording a JSON config,
      2) dataset building (two-view pipeline),
      3) model construction (encoder + projector + VICRegTrainer),
      4) optimizer and callbacks (including optional cosine schedules),
      5) training with `.fit()`,
      6) saving both full and encoder-only weights.
    """
    args = parse_args()

    # Use float32 by default. I can flip to bfloat16 via set_mixed_precision(True) if desired.
    set_mixed_precision(False)

    # Build infinite two-view dataset and report steps/epoch.
    ds = build_dataset(args.dataset, args.image_size, args.batch_size)
    _, steps_per_epoch = steps_for_dataset(args.dataset, args.batch_size)
    print(
        f"[train] dataset={args.dataset} img={args.image_size} "
        f"bs={args.batch_size} epochs={args.epochs} steps/epoch={steps_per_epoch}"
    )

    # Decide where to write artifacts (weights/config/metrics).
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M")
    if args.ckpt_out:
        # If an explicit weights path is provided, I co-locate side artifacts next to it.
        os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
        out_dir = os.path.dirname(args.ckpt_out)
        full_ckpt = args.ckpt_out
        enc_ckpt = full_ckpt.replace(".weights.h5", ".encoder.weights.h5")
    else:
        base = args.model_dir
        run_prefix = args.run_name or "pretrain"
        out_dir = os.path.join(base, f"{run_prefix}_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        full_ckpt = os.path.join(out_dir, "vicreg_full.weights.h5")
        enc_ckpt = os.path.join(out_dir, "vicreg_encoder.weights.h5")

    # Persist the config for reproducibility.
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

    device_str = decide_device(_pre_args.device)
    print(f"[train] Using device scope: {device_str}")

    with tf.device(device_str):
        # 1) Build the encoder and projector.
        encoder = build_encoder(args.image_size, feat_dim=args.feat_dim)
        projector = build_projector(args.feat_dim, args.proj_out, args.proj_layers)

        # 2) Optimizer: prefer AdamW (tfa) if available, else Adam.
        try:
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.wd)
        except Exception:
            opt = keras.optimizers.Adam(learning_rate=args.lr)

        # 3) Wrap encoder+projector into VICRegTrainer (my subclassed Model).
        trainer = VICRegTrainer(
            encoder=encoder,
            projector=projector,
            w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
            adaptive=args.adaptive,
            use_schedules=args.use_schedules,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            base_lr=args.lr,           # <-- NEW
            base_wd=args.wd,           # <-- NEW
        )
        trainer.compile(optimizer=opt)

        # 4) Build variables so save_weights works for subclassed models.
        force_build_for_saving(trainer, encoder, projector, args.image_size)

        # 5) Keras callbacks: save best weights by total loss, stop on NaN.
        ckpt_cb = keras.callbacks.ModelCheckpoint(
            filepath=full_ckpt,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
        ton_cb = keras.callbacks.TerminateOnNaN()

        # 6) Internal metrics callback (this powers my report_metrics plots/tables).
        sample_images = _take_single_view_batch(ds, size_limit=args.metrics_probe_batch)
        metrics_cb = VicRegMetricsLogger(
            run_dir=out_dir,
            encoder=encoder,
            projector=projector,
            sample_images=sample_images,
            compute_on=args.metrics_compute_on,
            loss_keys={"total": "loss", "align": "l_align", "var": "l_var", "cov": "l_cov"},
            record_every=args.record_every,
        )

        # 7) Optional cosine schedule (applied per optimizer step).
        cbs: list[keras.callbacks.Callback] = [ckpt_cb, ton_cb, metrics_cb]
        if args.use_schedules:
            total_steps = steps_per_epoch * args.epochs
            cbs.insert(0, CosineScheduleCallback(optimizer=opt,
                                                 total_steps=total_steps,
                                                 base_lr=args.lr,
                                                 base_wd=args.wd,
                                                 verbose=1))

        # 8) Train. The dataset repeats; steps_per_epoch bounds each epoch.
        trainer.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=cbs,
            verbose=1,
        )

        # 9) Always save encoder-only snapshot for downstream evaluation.
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
