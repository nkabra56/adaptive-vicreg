"""
Report generator: plots and tables comparing VICReg runs (baseline vs.
adaptive, or any set of named runs) on CIFAR-10.

Produces, under --out-dir:
- Training-dynamics plots (loss components and embedding stats), overlaid
  across runs.
- table_linear.csv/.png: method, top1, top5, f1_weighted (from linear-probe CSVs).
- table_knn.csv/.png: method, top1_k20, top1_k200 (from kNN eval CSVs, falling
  back to a k/temperature grid CSV if the eval CSV lacks explicit k columns).
- perf_vs_batchsize_k200.png: Top-1@k=200 (kNN) vs. training batch size, when
  at least two runs provide a train_config.json.

Usage (single run):
  python3 src/vicreg_tf/report_metrics.py \
    --out-dir "reports/${RUN_NAME}" \
    --history "name=AdaptiveVICReg,path=${RUN_DIR}/metrics/history.jsonl" \
    --linear-csv "results/${RUN_NAME}/${RUN_NAME}_linear_adamw.csv" \
    --knn-csv    "results/${RUN_NAME}/${RUN_NAME}_knn_eval.csv"

Usage (two runs, with auto-overlays and robustness plot):
  python3 src/vicreg_tf/report_metrics.py \
    --out-dir "reports/COMPARE_baseline_vs_adaptive" \
    --history "name=VICReg,path=checkpoints_tf/baseline_run/metrics/history.jsonl,config=checkpoints_tf/baseline_run/train_config.json" \
    --history "name=AdaptiveVICReg,path=checkpoints_tf/adaptive_run/metrics/history.jsonl,config=checkpoints_tf/adaptive_run/train_config.json" \
    --linear-csv "name=VICReg,path=results/baseline_run/baseline_run_linear_adamw.csv" \
    --linear-csv "name=AdaptiveVICReg,path=results/adaptive_run/adaptive_run_linear_adamw.csv" \
    --knn-csv "name=VICReg,path=results/baseline_run/baseline_run_knn_eval.csv" \
    --knn-csv "name=AdaptiveVICReg,path=results/adaptive_run/adaptive_run_knn_eval.csv" \
    --knn-grid "name=VICReg,path=results/baseline_run/baseline_run_knn_grid.csv" \
    --knn-grid "name=AdaptiveVICReg,path=results/adaptive_run/adaptive_run_knn_grid.csv"

Notes:
- If 'config=' is omitted from --history, train_config.json is inferred from
  the history path.
- Linear CSVs: missing top5/f1_weighted columns show as NaN, not a failure.
- kNN: if k=20/200 columns are missing from the eval CSV, the grid CSV (if
  provided) is mined for the best row at k=20 and k=200.
- stats/avg_std is plotted with a horizontal gamma=1 line (the baseline
  target); if a history logs 'target/gamma' per epoch, it's overlaid as a
  dashed curve.
- stats/avg_offdiag_corr_sq is plotted on a log y-axis to visualize
  decorrelation progress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Parsing helpers ---------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_kv_spec(spec: str) -> Dict[str, str]:
    """
    Parse "key=val,key=val,..." into a dict. A bare value with no "path="
    prefix (e.g. "/a/b/linears.csv") is treated as the path.
    """
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            if "path" not in out:
                out["path"] = part
    return out


@dataclass
class RunEntry:
    """One run: its history/config paths plus attached linear/kNN CSVs."""
    name: str
    history: str
    config: Optional[str] = None
    linear_csvs: List[str] = None
    knn_csvs: List[str] = None
    knn_grid_csvs: List[str] = None

    def __post_init__(self):
        self.linear_csvs = self.linear_csvs or []
        self.knn_csvs = self.knn_csvs or []
        self.knn_grid_csvs = self.knn_grid_csvs or []

    def infer_config_from_history(self) -> None:
        """If `config` wasn't given, look for ../train_config.json relative to the history path."""
        if self.config and os.path.exists(self.config):
            return
        try:
            metrics_dir = os.path.dirname(self.history)
            run_dir = os.path.dirname(metrics_dir)
            guess = os.path.join(run_dir, "train_config.json")
            if os.path.exists(guess):
                self.config = guess
        except Exception:
            pass


def _attach_file_to_named_run(runs_by_name: Dict[str, RunEntry],
                              spec_list: List[str],
                              slot: str) -> None:
    """
    Attach linear/knn/knn-grid file paths to runs. Each spec is either
    "name=Foo,path=/x/y.csv" or a bare path, which is attached to the only
    run if there's exactly one.
    """
    if not spec_list:
        return
    unnamed_targets: List[str] = []
    for raw in spec_list:
        kv = _parse_kv_spec(raw)
        name = kv.get("name")
        path = kv.get("path")
        if not path:
            continue
        if name:
            runs_by_name.setdefault(name, RunEntry(name=name, history="")).__dict__[slot].append(path)
        else:
            unnamed_targets.append(path)

    if unnamed_targets and len(runs_by_name) == 1:
        only = next(iter(runs_by_name.values()))
        only.__dict__[slot].extend(unnamed_targets)


# --- I/O + safe readers -------------------------------------------------------

def read_history_jsonl(path: str) -> Optional[pd.DataFrame]:
    """Load history.jsonl into a DataFrame sorted by epoch, or None if missing."""
    if not path or not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "epoch" not in df.columns:
        df["epoch"] = np.arange(1, len(df) + 1)
    else:
        df["epoch"] = df["epoch"].astype(int)

    keep = [
        "epoch",
        "loss/total", "loss/align", "loss/var", "loss/cov",
        "stats/avg_std", "stats/avg_offdiag_corr_sq",
        "target/gamma",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].sort_values("epoch")


def read_train_config(path: Optional[str]) -> Dict:
    """Load train_config.json, or {} if missing or unreadable."""
    try:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def read_csv_soft(path: str) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists and is parseable, else None."""
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# --- Metric selectors / reducers ----------------------------------------------

def _pick_acc_col(df: pd.DataFrame, prefer_k200: bool = False) -> Optional[str]:
    """Pick an accuracy column (name containing 'top1'/'acc'); prefer one mentioning '200' if `prefer_k200`."""
    if df is None:
        return None
    cands = [c for c in df.columns if ("top1" in c.lower()) or (c.lower() == "acc") or ("val_acc" in c.lower())]
    if not cands:
        return None
    if prefer_k200:
        spec = [c for c in cands if "200" in c]
        if spec:
            return spec[0]
    return cands[0]


def _best_value(df: pd.DataFrame, col: str) -> Optional[float]:
    """Max value of `col` in `df`, or None on failure."""
    try:
        return float(df[col].max())
    except Exception:
        return None


def _last_value(df: pd.DataFrame, col: str) -> Optional[float]:
    """Last value of `col` in `df`, or None on failure."""
    try:
        return float(df[col].iloc[-1])
    except Exception:
        return None


def _knn_best_at_k_from_grid(grid_df: pd.DataFrame, k: int) -> Optional[float]:
    """From a kNN grid CSV (columns include 'k' and an accuracy column), best top-1 at a fixed k across temperatures."""
    if grid_df is None or "k" not in grid_df.columns:
        return None
    dfk = grid_df[grid_df["k"] == k]
    if dfk.empty:
        return None
    col = _pick_acc_col(dfk, prefer_k200=(k == 200))
    if not col:
        return None
    return _best_value(dfk, col)


# --- Plotting utils ------------------------------------------------------------

def _savefig(out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_histories_overlaid(out_dir: str, runs: List[Tuple[str, pd.DataFrame]], key: str, title: str,
                            logy: bool = False, hline: Optional[Tuple[float, str]] = None,
                            extra_series: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None) -> Optional[str]:
    """
    Overlay a metric across runs.

    Args:
        runs: (run_name, history_df) pairs.
        key: History column to plot (e.g. 'loss/var' or 'stats/avg_std').
        hline: Optional (y, label) horizontal reference line (e.g. gamma=1).
        extra_series: Optional (label, epochs, values) series to overlay,
            e.g. a logged dynamic target.
    """
    plotted = False
    plt.figure(figsize=(7.5, 4.3))
    for name, h in runs:
        if h is None or key not in h.columns:
            continue
        plt.plot(h["epoch"], h[key], label=name)
        plotted = True

    if not plotted and not extra_series and not hline:
        return None

    if extra_series:
        for label, xs, ys in extra_series:
            plt.plot(xs, ys, linestyle="--", label=label)

    if hline:
        y, label = hline
        plt.axhline(y, linestyle=":", color="gray", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    if logy:
        ymin, ymax = plt.ylim()
        plt.ylim(max(ymin, 1e-8), ymax)  # avoid log(0)
        plt.yscale("log")
    plt.legend()
    return _savefig(out_dir, f"{key.replace('/','_')}.png")


# --- Table image helpers --------------------------------------------------------

def _format_numbers_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """Copy of `df` with numeric columns rounded and stringified; NaNs become empty strings."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{decimals}f}")
    return out


def _save_table_image(df: pd.DataFrame, out_dir: str, filename: str, title: Optional[str] = None,
                      header_facecolor: str = "#f0f0f0", fontsize: int = 10) -> str:
    """Render `df` as a PNG table, sized from its row/column counts."""
    _ensure_dir(out_dir)
    df_disp = _format_numbers_for_display(df)
    n_rows, n_cols = df_disp.shape

    col_w = 1.6
    row_h = 0.5
    fig_w = max(4.0, min(20.0, 1.2 + col_w * n_cols))
    fig_h = max(2.2, min(30.0, 1.0 + row_h * (n_rows + 1)))  # +1 for header

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=12)

    table = ax.table(
        cellText=df_disp.values,
        colLabels=list(df_disp.columns),
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_facecolor)
            cell.set_text_props(weight="bold")

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out_path


# --- Tables ----------------------------------------------------------------------

def build_linear_table(out_dir: str, run_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Build the linear-probe table (method, top1, top5, f1_weighted), writing
    table_linear.csv and table_linear.png under `out_dir`. `top5` and
    `f1_weighted` are optional columns; missing values are left as None.
    """
    rows = []
    for name, paths in run_map.items():
        # Prefer an AdamW linear CSV if both exist, else the first available.
        pref = sorted(paths, key=lambda p: (0 if "adam" in os.path.basename(p).lower() else 1, p))
        chosen = None
        for p in pref:
            if os.path.exists(p):
                chosen = p; break
        if not chosen:
            continue

        df = read_csv_soft(chosen)
        if df is None or df.empty:
            continue

        top1_col = _pick_acc_col(df, prefer_k200=False)
        top1 = _best_value(df, top1_col) if top1_col else None
        top5 = _best_value(df, "top5") if "top5" in df.columns else None
        f1w  = _best_value(df, "f1_weighted") if "f1_weighted" in df.columns else None

        method = None
        if "method" in df.columns:
            try:
                method = str(df["method"].iloc[0])
            except Exception:
                pass
        method = method or name

        rows.append({
            "method": method,
            "top1": top1,
            "top5": top5,
            "f1_weighted": f1w,
            "source_csv": os.path.relpath(chosen)
        })

    out = pd.DataFrame(rows)
    cols = [c for c in ["method", "top1", "top5", "f1_weighted", "source_csv"] if c in out.columns]
    out = out[cols] if cols else out

    csv_path = os.path.join(out_dir, "table_linear.csv")
    out.to_csv(csv_path, index=False)
    _save_table_image(out, out_dir, "table_linear.png", title="Linear Probe (CIFAR-10)")
    return out


def build_knn_table(out_dir: str,
                    eval_map: Dict[str, List[str]],
                    grid_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Build the kNN table (method, top1_k20, top1_k200), writing table_knn.csv
    and table_knn.png under `out_dir`. Reads *_knn_eval.csv first; if it
    lacks explicit k=20/k=200 columns, falls back to mining *_knn_grid.csv
    for the best accuracy at each k across temperatures.
    """
    rows = []

    all_names = set(eval_map.keys()).union(set(grid_map.keys()))
    for name in sorted(all_names):
        top1_k20 = None
        top1_k200 = None
        src_eval = None

        eval_paths = eval_map.get(name, [])
        grid_paths = grid_map.get(name, [])

        if eval_paths:
            p = [x for x in eval_paths if os.path.exists(x)]
            if p:
                src_eval = p[-1]  # most recent
                df = read_csv_soft(src_eval)
                if df is not None and not df.empty:
                    if "top1_k20" in df.columns:
                        top1_k20 = _last_value(df, "top1_k20")
                    if "top1_k200" in df.columns:
                        top1_k200 = _last_value(df, "top1_k200")
                    if (top1_k20 is None or top1_k200 is None) and "k" in df.columns:
                        col = _pick_acc_col(df, prefer_k200=False)
                        if col:
                            if top1_k20 is None:
                                best = df[df["k"] == 20]
                                if not best.empty:
                                    top1_k20 = _best_value(best, col)
                            if top1_k200 is None:
                                best = df[df["k"] == 200]
                                if not best.empty:
                                    top1_k200 = _best_value(best, col)

        if (top1_k20 is None or top1_k200 is None) and grid_paths:
            p = [x for x in grid_paths if os.path.exists(x)]
            if p:
                df = read_csv_soft(p[-1])
                if df is not None and not df.empty:
                    if top1_k20 is None:
                        top1_k20 = _knn_best_at_k_from_grid(df, 20)
                    if top1_k200 is None:
                        top1_k200 = _knn_best_at_k_from_grid(df, 200)

        method = name
        if src_eval:
            df = read_csv_soft(src_eval)
            if df is not None and "method" in df.columns:
                try:
                    method = str(df["method"].iloc[0]) or method
                except Exception:
                    pass

        rows.append({
            "method": method,
            "top1_k20": top1_k20,
            "top1_k200": top1_k200,
            "source_eval_csv": os.path.relpath(src_eval) if src_eval else ""
        })

    out = pd.DataFrame(rows)
    cols = [c for c in ["method", "top1_k20", "top1_k200", "source_eval_csv"] if c in out.columns]
    out = out[cols] if cols else out

    csv_path = os.path.join(out_dir, "table_knn.csv")
    out.to_csv(csv_path, index=False)
    _save_table_image(out, out_dir, "table_knn.png", title="kNN Evaluation (CIFAR-10)")
    return out


# --- Robustness (batch size) -------------------------------------------------------

def plot_perf_vs_batchsize_k200(out_dir: str,
                                runs_with_cfg: List[Tuple[str, Dict]],
                                knn_by_name: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Optional[str]:
    """
    Plot Top-1@k=200 (kNN) vs. training batch size, one point per run. For
    each run, reads `batch_size` from train_config.json and the k=200
    accuracy from its kNN eval or grid CSV. Returns the saved path, or None
    if no run has both values.
    """
    xs, ys, labels = [], [], []
    for name, cfg in runs_with_cfg:
        bs = cfg.get("batch_size")
        if not isinstance(bs, int):
            continue

        eval_df, grid_df = knn_by_name.get(name, (None, None))
        val = None
        if eval_df is not None:
            if "top1_k200" in eval_df.columns:
                val = _last_value(eval_df, "top1_k200")
            else:
                if "k" in eval_df.columns:
                    col = _pick_acc_col(eval_df, prefer_k200=True)
                    if col:
                        best = eval_df[eval_df["k"] == 200]
                        if not best.empty:
                            val = _best_value(best, col)
        if val is None and grid_df is not None:
            val = _knn_best_at_k_from_grid(grid_df, 200)
        if val is None:
            continue

        xs.append(bs); ys.append(val); labels.append(name)

    if not xs:
        return None

    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    labels = np.array(labels)[order]

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(xs, ys, marker="o")
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    plt.xlabel("Batch Size")
    plt.ylabel("Top-1 Accuracy @ k=200 (%)")
    plt.title("Performance vs. Batch Size (kNN)")
    return _savefig(out_dir, "perf_vs_batchsize_k200.png")


# --- Main -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=str, help="Directory to write plots and tables.")
    ap.add_argument("--history", action="append", default=[],
                    help="Run spec 'name=...,path=/.../metrics/history.jsonl[,config=/.../train_config.json]'. "
                         "Repeatable, to overlay multiple runs. If 'config' is omitted, it's inferred from the history path.")
    ap.add_argument("--linear-csv", action="append", default=[],
                    help="(Optional) Linear probe CSV path or 'name=...,path=...'.")
    ap.add_argument("--knn-csv", action="append", default=[],
                    help="(Optional) kNN eval CSV path or 'name=...,path=...'.")
    ap.add_argument("--knn-grid", action="append", default=[],
                    help="(Optional) kNN grid CSV (k x temperature) path or 'name=...,path=...'.")
    args = ap.parse_args()

    out_dir = args.out_dir
    _ensure_dir(out_dir)

    runs_by_name: Dict[str, RunEntry] = {}

    for raw in args.history:
        kv = _parse_kv_spec(raw)
        name = kv.get("name")
        path = kv.get("path")
        cfg  = kv.get("config")
        if not name or not path:
            raise ValueError("Each --history must include at least name=...,path=...")
        runs_by_name[name] = RunEntry(name=name, history=path, config=cfg)

    _attach_file_to_named_run(runs_by_name, args.linear_csv, "linear_csvs")
    _attach_file_to_named_run(runs_by_name, args.knn_csv, "knn_csvs")
    _attach_file_to_named_run(runs_by_name, args.knn_grid, "knn_grid_csvs")

    for r in runs_by_name.values():
        r.infer_config_from_history()

    hist_map: Dict[str, pd.DataFrame] = {}
    cfg_map: Dict[str, Dict] = {}
    for name, r in runs_by_name.items():
        hist_map[name] = read_history_jsonl(r.history)
        cfg_map[name] = read_train_config(r.config)

    lin_map = {name: r.linear_csvs for name, r in runs_by_name.items()}
    df_linear = build_linear_table(out_dir, lin_map)

    knn_eval_map = {name: r.knn_csvs for name, r in runs_by_name.items()}
    knn_grid_map = {name: r.knn_grid_csvs for name, r in runs_by_name.items()}
    df_knn = build_knn_table(out_dir, knn_eval_map, knn_grid_map)

    named_histories = [(name, hist_map[name]) for name in runs_by_name.keys()]

    plot_histories_overlaid(out_dir, named_histories, "loss/total", "Training Loss (total)")
    plot_histories_overlaid(out_dir, named_histories, "loss/align", "Alignment Loss")
    plot_histories_overlaid(out_dir, named_histories, "loss/var", "Variance Loss")
    plot_histories_overlaid(out_dir, named_histories, "loss/cov", "Covariance Loss")

    # avg_std: draw the gamma=1 baseline target, and overlay a logged
    # per-epoch target/gamma curve if any history has one.
    extra_gamma_series = []
    for name, h in named_histories:
        if h is not None and "target/gamma" in h.columns:
            extra_gamma_series.append((f"{name}: gamma_t", h["epoch"].values, h["target/gamma"].values))
    plot_histories_overlaid(out_dir, named_histories, "stats/avg_std", "Avg per-dim std (features)",
                            hline=(1.0, "gamma = 1.0 (baseline target)"),
                            extra_series=extra_gamma_series if extra_gamma_series else None)

    plot_histories_overlaid(out_dir, named_histories, "stats/avg_offdiag_corr_sq",
                            "Mean off-diagonal corr$^2$ (log scale)", logy=True)

    runs_with_cfg = [(name, cfg_map[name]) for name in runs_by_name.keys() if cfg_map[name]]

    knn_by_name: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, r in runs_by_name.items():
        eval_df = None
        if r.knn_csvs:
            last = [p for p in r.knn_csvs if os.path.exists(p)]
            if last:
                eval_df = read_csv_soft(last[-1])
        grid_df = None
        if r.knn_grid_csvs:
            last = [p for p in r.knn_grid_csvs if os.path.exists(p)]
            if last:
                grid_df = read_csv_soft(last[-1])
        knn_by_name[name] = (eval_df, grid_df)

    if len(runs_with_cfg) >= 2:
        plot_perf_vs_batchsize_k200(out_dir, runs_with_cfg, knn_by_name)

    print(f"[report] Wrote tables to: {out_dir}/table_linear.csv, {out_dir}/table_knn.csv")
    print(f"[report] Also saved images: {out_dir}/table_linear.png, {out_dir}/table_knn.png")
    print(f"[report] Plots saved under: {out_dir}")
    if len(runs_with_cfg) < 2:
        print("[report] Robustness plot (perf_vs_batchsize_k200.png) rendered only if >=2 runs with train_config.json were provided.")


if __name__ == "__main__":
    main()
