"""
Simple results aggregator / plotting utility.

Purpose
-------
Load JSON result files produced by:
  - scripts/eval_linear.py  (task == "linear_eval", metric: test_acc)
  - scripts/knn_eval.py     (task == "knn_eval",    metric: test_acc)

Then create a compact bar chart comparing runs.

Usage examples
--------------
# Summarize linear probe runs
python3 scripts/plots.py --glob "results/linear_eval_*.json" --out results/linear_summary.png

# Summarize kNN runs
python3 scripts/plots.py --glob "results/knn_eval_*.json" --out results/knn_summary.png

Notes
-----
- Uses matplotlib in headless mode (Agg backend).
- Avoids seaborn and style packs to keep dependencies minimal.

Author: Nishant Kabra
Date: 11/10/25
"""
from __future__ import annotations
import os
import glob
import json
import argparse

import matplotlib
matplotlib.use("Agg")  # headless safe (no GUI needed)
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Read CLI flags for which files to load and where to save the plot.

    Returns
    -------
    argparse.Namespace
        --glob : glob pattern matching result JSONs
        --out  : output PNG path
    """
    p = argparse.ArgumentParser()
    p.add_argument("--glob", type=str, required=True, help="Glob pattern for JSON results files.")
    p.add_argument("--out", type=str, required=True, help="Output image path (PNG).")
    return p.parse_args()


def load_results(paths):
    """
    Read all JSON files into memory.

    Parameters
    ----------
    paths : list[str]
        Files matched by the glob.

    Returns
    -------
    list[tuple[str, dict]]
        (path, parsed_json) pairs. Files that fail to parse are skipped with a warning.
    """
    rows = []
    for p in paths:
        try:
            with open(p, "r") as f:
                rows.append((p, json.load(f)))
        except Exception as e:
            print("Failed to read", p, ":", repr(e))
    return rows


def label_from_result(r: dict) -> str:
    """
    Generate a short label per bar using task/dataset/proj sizes.

    Returns
    -------
    str
        Example: "linear-cifar10-o8192-L3" or "knn-cifar100-o8192-L3"
    """
    ds = r.get("dataset", "?")
    task = r.get("task", "?")
    proj_out = r.get("proj_out", "?")
    proj_layers = r.get("proj_layers", "?")
    if task == "linear_eval":
        return f"linear-{ds}-o{proj_out}-L{proj_layers}"
    if task == "knn_eval":
        return f"knn-{ds}-o{proj_out}-L{proj_layers}"
    return f"{task}-{ds}"


def value_from_result(r: dict) -> float:
    """
    Extract the plotted scalar (test accuracy) from a result JSON.

    Returns
    -------
    float
        Defaults to 0.0 if the key is missing.
    """
    return float(r.get("test_acc", 0.0))


def plot_bar(pairs, out_path: str) -> None:
    """
    Make a bar chart of accuracy vs run label and write a PNG file.

    Parameters
    ----------
    pairs : list[tuple[str, dict]]
        (path, json) pairs as returned by `load_results()`.
    out_path : str
        Where to save the PNG.

    Side Effects
    ------------
    Writes the PNG to `out_path`.
    """
    labels = [label_from_result(r) for _, r in pairs]
    vals = [value_from_result(r) for _, r in pairs]

    # Size scales with number of bars to keep labels readable.
    fig = plt.figure(figsize=(max(6, 1.2 * len(vals)), 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Evaluation summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plots] wrote {out_path}")


def main():
    """
    Entrypoint: glob -> load -> bar plot.
    """
    args = parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("[plots] No files matched:", args.glob)
        return
    pairs = load_results(paths)
    if not pairs:
        print("[plots] No readable JSON files.")
        return
    plot_bar(pairs, args.out)


if __name__ == "__main__":
    main()
