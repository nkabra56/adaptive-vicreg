"""Plots and tables that compare pretraining runs (for example baseline vs. adaptive) on CIFAR-10.

Writes these files to --out-dir:
- Overlaid training curves for each loss term, `stats/avg_std` and `stats/avg_offdiag_corr_sq`.
- table_linear.csv/.png: method, top1, top5, f1_weighted, from the linear-probe CSVs.
- table_knn.csv/.png: method, top1_k20, top1_k200, from the kNN CSVs, falling back to a k/temperature
  grid CSV when the eval CSV has no k columns.
- perf_vs_batchsize_k200.png: kNN top-1 at k=200 against training batch size, when at least two runs
  have a train_config.json.

Every input is a run spec like "name=VICReg,path=.../history.jsonl". Repeat a flag to add runs.

Example:
  python3 src/vicreg_tf/report_metrics.py \\
    --out-dir reports/compare \\
    --history "name=VICReg,path=checkpoints_tf/baseline/metrics/history.jsonl" \\
    --history "name=AdaptiveVICReg,path=checkpoints_tf/adaptive/metrics/history.jsonl" \\
    --linear-csv "name=VICReg,path=results/baseline/linear_eval.csv" \\
    --knn-csv "name=VICReg,path=results/baseline/knn_eval.csv"

Notes:
- Without `config=` in a --history spec, train_config.json is looked up next to the run's metrics folder.
- Missing top5 or f1_weighted columns show up as NaN.
- `stats/avg_std` is drawn with a gamma = 1 reference line, plus a dashed `target/gamma` curve if a history logs one.
- `stats/avg_offdiag_corr_sq` is drawn on a log axis.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_kv_spec(spec: str) -> Dict[str, str]:
    """Parse "key=val,key=val" into a dict. A bare value without a key is taken as the path."""
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
        elif "path" not in out:
            out["path"] = part
    return out


@dataclass
class RunEntry:
    """One run: its history and config paths, plus the linear and kNN CSVs attached to it."""
    name: str
    history: str
    config: Optional[str] = None
    linear_csvs: List[str] = field(default_factory=list)
    knn_csvs: List[str] = field(default_factory=list)
    knn_grid_csvs: List[str] = field(default_factory=list)

    def infer_config_from_history(self) -> None:
        """If no config was given, look for train_config.json in the run directory above the metrics folder."""
        if self.config and os.path.exists(self.config):
            return
        run_dir = os.path.dirname(os.path.dirname(self.history))
        guess = os.path.join(run_dir, "train_config.json")
        if os.path.exists(guess):
            self.config = guess


def _attach_file_to_named_run(runs_by_name: Dict[str, RunEntry], spec_list: List[str], slot: str) -> None:
    """Attach CSV paths from `spec_list` to the `slot` list of each named run.

    A spec without a name is attached to the only run, if there is exactly one.
    """
    unnamed: List[str] = []
    for raw in spec_list:
        kv = _parse_kv_spec(raw)
        name, path = kv.get("name"), kv.get("path")
        if not path:
            continue
        if name:
            getattr(runs_by_name.setdefault(name, RunEntry(name=name, history="")), slot).append(path)
        else:
            unnamed.append(path)

    if unnamed and len(runs_by_name) == 1:
        getattr(next(iter(runs_by_name.values())), slot).extend(unnamed)


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


def _pick_acc_col(df: pd.DataFrame, prefer_k200: bool = False) -> Optional[str]:
    """Pick the accuracy column: 'top1*', 'acc', 'test_acc' or 'val_acc'. With `prefer_k200`, prefer a k=200 one."""
    if df is None:
        return None
    cands = [
        c for c in df.columns
        if "top1" in c.lower() or c.lower() in ("acc", "test_acc") or "val_acc" in c.lower()
    ]
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
    """Best top-1 at a fixed `k` across temperatures, from a kNN grid CSV with a 'k' column."""
    if grid_df is None or "k" not in grid_df.columns:
        return None
    dfk = grid_df[grid_df["k"] == k]
    if dfk.empty:
        return None
    col = _pick_acc_col(dfk, prefer_k200=(k == 200))
    if not col:
        return None
    return _best_value(dfk, col)


def _savefig(out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_histories_overlaid(out_dir: str, runs: List[Tuple[str, pd.DataFrame]], key: str, title: str,
                            logy: bool = False, hline: Optional[Tuple[float, str]] = None,
                            extra_series: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None) -> Optional[str]:
    """Overlay one history column across runs and save it as `<key>.png`.

    Args:
        runs: (run_name, history_df) pairs.
        key: History column to plot, e.g. 'loss/var'.
        hline: Optional (y, label) reference line.
        extra_series: Optional (label, epochs, values) series drawn dashed.
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
    return _savefig(out_dir, f"{key.replace('/', '_')}.png")


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


def build_linear_table(out_dir: str, run_map: Dict[str, List[str]]) -> pd.DataFrame:
    """Write table_linear.csv/.png (method, top1, top5, f1_weighted) under `out_dir`.

    `top5` and `f1_weighted` are optional columns and stay empty when absent.
    """
    rows = []
    for name, paths in run_map.items():
        # Prefer an AdamW CSV when there are several.
        pref = sorted(paths, key=lambda p: (0 if "adam" in os.path.basename(p).lower() else 1, p))
        chosen = next((p for p in pref if os.path.exists(p)), None)
        if not chosen:
            continue

        df = read_csv_soft(chosen)
        if df is None or df.empty:
            continue

        top1_col = _pick_acc_col(df, prefer_k200=False)
        top1 = _best_value(df, top1_col) if top1_col else None
        top5 = _best_value(df, "top5") if "top5" in df.columns else None
        f1w = _best_value(df, "f1_weighted") if "f1_weighted" in df.columns else None

        method = (str(df["method"].iloc[0]) if "method" in df.columns else "") or name

        rows.append({
            "method": method,
            "top1": top1,
            "top5": top5,
            "f1_weighted": f1w,
            "source_csv": os.path.relpath(chosen),
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
    """Write table_knn.csv/.png (method, top1_k20, top1_k200) under `out_dir`.

    Values come from the kNN eval CSV first. Whatever it lacks is filled from the grid CSV, using the best
    accuracy at each k across temperatures.
    """
    rows = []

    for name in sorted(set(eval_map) | set(grid_map)):
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
            if df is not None and not df.empty and "method" in df.columns:
                method = str(df["method"].iloc[0]) or method

        rows.append({
            "method": method,
            "top1_k20": top1_k20,
            "top1_k200": top1_k200,
            "source_eval_csv": os.path.relpath(src_eval) if src_eval else "",
        })

    out = pd.DataFrame(rows)
    cols = [c for c in ["method", "top1_k20", "top1_k200", "source_eval_csv"] if c in out.columns]
    out = out[cols] if cols else out

    csv_path = os.path.join(out_dir, "table_knn.csv")
    out.to_csv(csv_path, index=False)
    _save_table_image(out, out_dir, "table_knn.png", title="kNN Evaluation (CIFAR-10)")
    return out


def plot_perf_vs_batchsize_k200(out_dir: str,
                                runs_with_cfg: List[Tuple[str, Dict]],
                                knn_by_name: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Optional[str]:
    """Plot kNN top-1 at k=200 against training batch size, one point per run.

    The batch size comes from train_config.json and the accuracy from the kNN eval or grid CSV.
    Returns the saved path, or None if no run has both.
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
            elif "k" in eval_df.columns:
                col = _pick_acc_col(eval_df, prefer_k200=True)
                if col:
                    best = eval_df[eval_df["k"] == 200]
                    if not best.empty:
                        val = _best_value(best, col)
        if val is None and grid_df is not None:
            val = _knn_best_at_k_from_grid(grid_df, 200)
        if val is None:
            continue

        xs.append(bs)
        ys.append(val)
        labels.append(name)

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Plots and tables that compare pretraining runs.")
    ap.add_argument("--out-dir", required=True, type=str, help="Directory to write plots and tables to.")
    ap.add_argument("--history", action="append", default=[],
                    help="Run spec 'name=...,path=.../history.jsonl[,config=.../train_config.json]'. "
                         "Repeat to add runs.")
    ap.add_argument("--linear-csv", action="append", default=[],
                    help="Linear-probe CSV, as a path or 'name=...,path=...'. Optional.")
    ap.add_argument("--knn-csv", action="append", default=[],
                    help="kNN eval CSV, as a path or 'name=...,path=...'. Optional.")
    ap.add_argument("--knn-grid", action="append", default=[],
                    help="kNN grid CSV (k by temperature), as a path or 'name=...,path=...'. Optional.")
    args = ap.parse_args()

    out_dir = args.out_dir
    _ensure_dir(out_dir)

    runs_by_name: Dict[str, RunEntry] = {}

    for raw in args.history:
        kv = _parse_kv_spec(raw)
        name, path, cfg = kv.get("name"), kv.get("path"), kv.get("config")
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

    build_linear_table(out_dir, {name: r.linear_csvs for name, r in runs_by_name.items()})
    build_knn_table(
        out_dir,
        {name: r.knn_csvs for name, r in runs_by_name.items()},
        {name: r.knn_grid_csvs for name, r in runs_by_name.items()},
    )

    named_histories = [(name, hist_map[name]) for name in runs_by_name]

    plot_histories_overlaid(out_dir, named_histories, "loss/total", "Training Loss (total)")
    plot_histories_overlaid(out_dir, named_histories, "loss/align", "Alignment Loss")
    plot_histories_overlaid(out_dir, named_histories, "loss/var", "Variance Loss")
    plot_histories_overlaid(out_dir, named_histories, "loss/cov", "Covariance Loss")

    extra_gamma_series = []
    for name, h in named_histories:
        if h is not None and "target/gamma" in h.columns:
            extra_gamma_series.append((f"{name}: gamma_t", h["epoch"].values, h["target/gamma"].values))
    plot_histories_overlaid(out_dir, named_histories, "stats/avg_std", "Avg per-dim std (features)",
                            hline=(1.0, "gamma = 1.0 (baseline target)"),
                            extra_series=extra_gamma_series if extra_gamma_series else None)

    plot_histories_overlaid(out_dir, named_histories, "stats/avg_offdiag_corr_sq",
                            "Mean off-diagonal corr$^2$ (log scale)", logy=True)

    runs_with_cfg = [(name, cfg_map[name]) for name in runs_by_name if cfg_map[name]]

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
        print("[report] perf_vs_batchsize_k200.png needs at least two runs with a train_config.json.")


if __name__ == "__main__":
    main()
