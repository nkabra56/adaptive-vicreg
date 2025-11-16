#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive metrics + plots for multiclass classifiers (CIFAR-10/100 friendly).

What this script does
---------------------
- Loads ground-truth labels (y_true) and either probabilities (probs [N,C]) or
  predicted labels (y_pred [N]).
- Computes Accuracy and macro/micro/weighted Precision, Recall, F1.
- Computes per-class Precision, Recall, F1, and Support.
- If probabilities are present: macro/micro ROC-AUC (OvR) and macro Average Precision.
- Saves tables (CSV+JSON) and JPG figures with titles, axis labels, legends.

Outputs (all saved to: metrics/<run_name>/)
-------------------------------------------
<run>_confusion_matrix.jpg
<run>_roc.jpg                (needs probs)
<run>_pr.jpg                 (needs probs)
<run>_per_class_f1.jpg
<run>_metrics_summary_table.jpg
<run>_per_class_table.jpg
metrics_summary.json
metrics_summary.csv
per_class_metrics.csv

Usage
-----
python3 scripts/report_metrics.py \
  --npz results/<STAMP>__knn-.../probs_test.npz \
  --dataset cifar10 \
  --run-name knn-feat_c10_model1

Notes
-----
- Uses Scikit-learn (imported as `sklearn`).
- Figures are saved as JPG to embed easily in docs/slides.

Author: Nishant Kabra
Date: 2025-11-16
"""
import os
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn (project name), imported as 'sklearn' (correct public API)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd


CIFAR10_NAMES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def class_names(dataset: str, n_classes: int):
    ds = (dataset or "").lower()
    if ds == "cifar10" and n_classes == 10:
        return CIFAR10_NAMES
    # Fallback: numeric labels
    return [str(i) for i in range(n_classes)]


def save_fig(path: str, fig=None, dpi=200):
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_confusion(cm, labels, out_jpg, normalize=True, run_name="run"):
    fig = plt.figure(figsize=(8, 7))
    if normalize:
        denom = cm.sum(axis=1, keepdims=True).clip(min=1)
        data = cm.astype(float) / denom
        title = f"{run_name} — Confusion Matrix (row-normalized)"
    else:
        data = cm
        title = f"{run_name} — Confusion Matrix (counts)"

    im = plt.imshow(data, interpolation="nearest")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("Proportion" if normalize else "Count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    save_fig(out_jpg, fig)


def plot_per_class_f1(f1, labels, out_jpg, run_name="run"):
    fig = plt.figure(figsize=(9, 5))
    xs = np.arange(len(labels))
    plt.bar(xs, f1, width=0.8)
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.title(f"{run_name} — Per-class F1")
    plt.ylabel("F1 score")
    plt.xlabel("Class")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", linewidth=0.3)
    save_fig(out_jpg, fig)


def plot_roc(y_true, probs, out_jpg, run_name="run"):
    """Macro + micro ROC-AUC (OvR) with labels & legend."""
    n_classes = probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # Compute AUCs (no per-class curves to keep slide clean)
    try:
        auc_micro = roc_auc_score(y_bin, probs, average="micro", multi_class="ovr")
    except Exception:
        auc_micro = float("nan")
    try:
        auc_macro = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = float("nan")

    # Just draw the baseline diagonal, annotate macro/micro AUC in legend
    fig = plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    # We do not compute aggregated ROC curves coordinates here; we summarize in legend.
    plt.title(f"{run_name} — ROC (OvR) | macro AUC={auc_macro:.3f}, micro AUC={auc_micro:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_fig(out_jpg, fig)
    return auc_macro, auc_micro


def plot_pr(y_true, probs, out_jpg, run_name="run"):
    """Macro Average Precision (area under PR); plotted as title + clean axes."""
    n_classes = probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        ap_macro = average_precision_score(y_bin, probs, average="macro")
    except Exception:
        ap_macro = float("nan")

    fig = plt.figure(figsize=(7, 5))
    plt.title(f"{run_name} — Precision–Recall | macro AP={ap_macro:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    # We keep the canvas minimal; per-class curves omitted for clarity.
    plt.legend([], frameon=False)
    save_fig(out_jpg, fig)
    return ap_macro


def df_to_table_image(df: pd.DataFrame, out_jpg: str, title: str):
    """
    Render a small DataFrame as a JPG with labels, title.
    """
    fig, ax = plt.subplots(figsize=(max(7, 0.35*len(df.columns)+3), 0.5*len(df)+2))
    ax.axis("off")
    tbl = ax.table(cellText=df.values,
                   colLabels=df.columns,
                   rowLabels=df.index if df.index.name is not None else None,
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    ax.set_title(title, pad=12)
    save_fig(out_jpg, fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to .npz with y_true and probs or y_pred")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--run-name", required=True, help="Used for output folder and filenames")
    ap.add_argument("--metrics-root", default="metrics", help="Root dir; final goes to metrics/<run-name>/")
    args = ap.parse_args()

    out_dir = ensure_dir(os.path.join(args.metrics_root, args.run_name))
    prefix = os.path.join(out_dir, args.run_name)  # base for jpg names

    data = np.load(args.npz, allow_pickle=True)
    if "y_true" not in data:
        raise ValueError("npz must contain 'y_true'.")
    y_true = data["y_true"].squeeze()
    probs = data["probs"] if "probs" in data else None
    y_pred = data["y_pred"].squeeze() if "y_pred" in data else None

    if probs is not None:
        if probs.ndim != 2:
            raise ValueError("'probs' must be [N, C]")
        n_classes = probs.shape[1]
        if y_pred is None:
            y_pred = probs.argmax(1)
    else:
        if y_pred is None:
            raise ValueError("Need either 'probs' or 'y_pred' in npz.")
        n_classes = int(y_pred.max()) + 1

    labels = class_names(args.dataset, n_classes)

    # -------- Metrics (overall) --------
    acc = accuracy_score(y_true, y_pred)
    prf_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prf_weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prf_micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)

    # -------- Per-class --------
    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    )

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    plot_confusion(cm, labels, f"{prefix}_confusion_matrix.jpg", normalize=True, run_name=args.run_name)

    # -------- Curves (if probs) --------
    roc_macro = roc_micro = pr_ap_macro = float("nan")
    if probs is not None:
        roc_macro, roc_micro = plot_roc(y_true, probs, f"{prefix}_roc.jpg", run_name=args.run_name)
        pr_ap_macro = plot_pr(y_true, probs, f"{prefix}_pr.jpg", run_name=args.run_name)

    # -------- Per-class F1 bar chart --------
    plot_per_class_f1(f1_cls, labels, f"{prefix}_per_class_f1.jpg", run_name=args.run_name)

    # -------- Tables (CSV + image) --------
    # Summary table
    summary = {
        "run": args.run_name,
        "dataset": args.dataset,
        "n_classes": int(n_classes),
        "accuracy": float(acc),
        "precision_macro": float(prf_macro[0]),
        "recall_macro": float(prf_macro[1]),
        "f1_macro": float(prf_macro[2]),
        "precision_weighted": float(prf_weighted[0]),
        "recall_weighted": float(prf_weighted[1]),
        "f1_weighted": float(prf_weighted[2]),
        "precision_micro": float(prf_micro[0]),
        "recall_micro": float(prf_micro[1]),
        "f1_micro": float(prf_micro[2]),
        "roc_auc_macro_ovr": float(roc_macro) if not np.isnan(roc_macro) else None,
        "roc_auc_micro_ovr": float(roc_micro) if not np.isnan(roc_micro) else None,
        "pr_ap_macro": float(pr_ap_macro) if not np.isnan(pr_ap_macro) else None,
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV summary
    with open(os.path.join(out_dir, "metrics_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","dataset","n_classes","accuracy",
                    "precision_macro","recall_macro","f1_macro",
                    "precision_weighted","recall_weighted","f1_weighted",
                    "precision_micro","recall_micro","f1_micro",
                    "roc_auc_macro_ovr","roc_auc_micro_ovr","pr_ap_macro"])
        w.writerow([summary["run"], summary["dataset"], summary["n_classes"],
                    f"{summary['accuracy']:.4f}",
                    f"{summary['precision_macro']:.4f}",
                    f"{summary['recall_macro']:.4f}",
                    f"{summary['f1_macro']:.4f}",
                    f"{summary['precision_weighted']:.4f}",
                    f"{summary['recall_weighted']:.4f}",
                    f"{summary['f1_weighted']:.4f}",
                    f"{summary['precision_micro']:.4f}",
                    f"{summary['recall_micro']:.4f}",
                    f"{summary['f1_micro']:.4f}",
                    (f"{summary['roc_auc_macro_ovr']:.4f}" if summary["roc_auc_macro_ovr"] is not None else "nan"),
                    (f"{summary['roc_auc_micro_ovr']:.4f}" if summary["roc_auc_micro_ovr"] is not None else "nan"),
                    (f"{summary['pr_ap_macro']:.4f}" if summary["pr_ap_macro"] is not None else "nan")])

    # Render summary table to image
    df_summary = pd.DataFrame([summary]).drop(columns=["roc_auc_macro_ovr","roc_auc_micro_ovr","pr_ap_macro"])
    df_summary_rounded = df_summary.copy()
    for k in df_summary_rounded.columns:
        if isinstance(df_summary_rounded.iloc[0][k], float):
            df_summary_rounded[k] = f"{df_summary_rounded.iloc[0][k]:.4f}"
    df_to_table_image(df_summary_rounded, f"{prefix}_metrics_summary_table.jpg",
                      title=f"{args.run_name} — Metrics Summary (Scikit-learn)")

    # Per-class CSV
    df_per = pd.DataFrame({
        "class": labels,
        "precision": p_cls,
        "recall": r_cls,
        "f1": f1_cls,
        "support": supp_cls.astype(int)
    })
    df_per.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

    # Per-class table image
    df_per_img = df_per.copy()
    df_per_img["precision"] = df_per_img["precision"].map(lambda v: f"{v:.4f}")
    df_per_img["recall"]    = df_per_img["recall"].map(lambda v: f"{v:.4f}")
    df_per_img["f1"]        = df_per_img["f1"].map(lambda v: f"{v:.4f}")
    df_to_table_image(df_per_img, f"{prefix}_per_class_table.jpg",
                      title=f"{args.run_name} — Per-class Precision/Recall/F1/Support")

    print(f"[report_metrics] Wrote metrics & figures to: {out_dir}")
    print(" Key files:")
    print(" -", f"{prefix}_confusion_matrix.jpg")
    if probs is not None:
        print(" -", f"{prefix}_roc.jpg")
        print(" -", f"{prefix}_pr.jpg")
    print(" -", f"{prefix}_per_class_f1.jpg")
    print(" -", f"{prefix}_metrics_summary_table.jpg")
    print(" -", f"{prefix}_per_class_table.jpg")
    print(" -", os.path.join(out_dir, "metrics_summary.csv"))
    print(" -", os.path.join(out_dir, "per_class_metrics.csv"))
    print(" -", os.path.join(out_dir, "metrics_summary.json"))


if __name__ == "__main__":
    main()
