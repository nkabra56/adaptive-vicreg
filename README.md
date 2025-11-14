# Adaptive VICReg (TensorFlow/Keras)

**Author:** Nishant Kabra  
**Date:** 11/10/2025

This repository contains a practical TensorFlow/Keras implementation of **VICReg** — Variance-Invariance-Covariance Regularization — with two enhancements designed for stability and performance on modest hardware:

1. **Adaptive Variance Targeting (AVT):** replaces the fixed variance floor $\gamma$ with a data-driven target estimated from an exponential moving average of per-dimension standard deviations (median used for robustness).
2. **Scale-Invariant Covariance (SICov):** normalizes covariance by its trace and penalizes the Frobenius distance to $(1/d)I$, making the redundancy term less sensitive to global feature scale.

The codebase includes **self-supervised pretraining**, **linear probing**, and **k-NN evaluation**, along with quality-of-life features (mixed precision, BN adaptation, robust checkpoint loading, and CPU-friendly defaults).

> **Reference (original method):**  
> Adrien Bardes, Jean Ponce, Yann LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** arXiv:2105.04906. PDF: https://arxiv.org/pdf/2105.04906

---

## Table of Contents

1. [What is VICReg?](#what-is-vicreg)  
2. [What's in this repo](#whats-in-this-repo)  
3. [Environment Setup](#environment-setup)  
4. [Datasets](#datasets)  
5. [Self-Supervised Pretraining](#self-supervised-pretraining)  
6. [Evaluation](#evaluation)  
   - [Linear Probe](#linear-probe)  
   - [k-NN Evaluation](#k-nn-evaluation)  
7. [Key Implementation Details](#key-implementation-details)  
   - [Data & Augmentations](#data--augmentations)  
   - [Backbone & Projector](#backbone--projector)  
   - [Losses](#losses)  
   - [Schedules](#schedules)  
   - [BN Adaptation](#bn-adaptation)  
   - [Mixed Precision on CPU/GPU](#mixed-precision-on-cpugpu)  
8. [Recommended Hyperparameters](#recommended-hyperparameters)  
9. [Troubleshooting & Gotchas](#troubleshooting--gotchas)  
10. [Reproducibility Tips](#reproducibility-tips)  
11. [Citations](#citations)  
12. [License](#license)  

---

## What is VICReg?

VICReg is a self-supervised learning objective defined over two differently augmented “views” of the same image. It encourages:

- **Invariance (Alignment):** matched features for the two views  
- **Variance:** per-dimension standard deviation above a floor \(\gamma\) to avoid collapse  
- **Covariance Decorrelation:** penalize off-diagonal covariance entries to reduce redundancy

Formally for batch features $z_1, z_2 \in \mathbb{R}^{B\times d}$:  
- Alignment: $\mathcal{L}_{\text{align}} = \frac{1}{B} \sum_i \lVert z_{1,i} - z_{2,i} \rVert_2^2$  
- Variance hinge: encourage $\text{std}(z_{\cdot,j}) \ge \gamma$ per dimension $j$  
- Covariance: sum of off-diagonal squared entries of the empirical covariance matrix

This repo keeps the spirit of VICReg and adds **AVT** and **SICov** to reduce manual tuning and make the objective less sensitive to global feature scaling.

---

## What’s in this repo

```
adaptive-vicreg-tf/
├─ scripts/
│  ├─ train_vicreg.py        # self-supervised pretraining (VICReg + AVT + SICov)
│  ├─ eval_linear.py         # linear probe on frozen encoder
│  └─ knn_eval.py            # k-NN accuracy using cosine similarity
├─ src/                      # (if using the packaged API, optional)
│  └─ vicreg_tf/             # augmentation, losses, schedules, model helpers
├─ checkpoints_tf/           # saved weights (e.g., vicreg_tf.weights.h5) or SavedModel dirs
├─ artifacts/                # exported encoder SavedModel for evaluation scripts
├─ plots.py                  # optional diagnostics/visualization utilities
├─ requirements.txt
└─ README.md
```

> You may have only the **scripts/** and top-level files if you’re using the “single-script” workflow. The evaluation scripts handle both **Keras weights (.h5)** and **SavedModel** directories.

---

## Environment Setup

> **Python:** 3.9–3.12 are commonly used.  
> **TensorFlow:** CPU-only works; GPU requires a matching **tensorflow** wheel and **CUDA + cuDNN**.

Create a virtual environment and install requirements:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

**CPU performance tips (optional):**  
- You can disable oneDNN fused ops if you see numerical diffs:
  ```bash
  export TF_ENABLE_ONEDNN_OPTS=0
  ```
- Restrict threads on noisy laptops:
  ```bash
  export TF_NUM_INTRAOP_THREADS=4
  export TF_NUM_INTEROP_THREADS=4
  ```

**GPU setup (optional):**  
Install CUDA + cuDNN that match your TensorFlow wheel. If TF cannot find CUDA, the scripts fall back to CPU and print a clear message. Mixed precision will use **float16** on GPUs and **bfloat16** on CPUs that support it.

---

## Datasets

Supported out of the box:

1. **CIFAR-10 / CIFAR-100** via `tf.keras.datasets`  
2. **STL-10** via `tensorflow_datasets` (set `--tfds-data-dir` to a local cache if needed)  
3. **Folder dataset** structured as:
   ```text
   /path/to/data_root/
     class_1/*.jpg
     class_2/*.jpg
     ...
   ```
   Self-supervised pretraining ignores labels; supervised probe uses them.

**Image sizes**: CIFAR uses **32**, STL-10 uses **96**, generic folders often use **224**. Choose smaller sizes on CPU to avoid OOM.

---

## Self-Supervised Pretraining

Example (CIFAR-10, CPU friendly, mixed bfloat16 on capable CPUs):

```bash
python3 scripts/train_vicreg.py \
  --device auto \
  --dataset cifar10 --image-size 32 \
  --epochs 100 --batch-size 256 \
  --adaptive --use-schedules \
  --proj-out 8192 --proj-layers 3
```

Key outputs:
- **Checkpoints:** `checkpoints_tf/` (Keras weights `.weights.h5` and/or SavedModel directory)  
- **Exported encoder:** `artifacts/encoder_savedmodel/` (used by eval scripts)

> If you see out-of-memory on CPU, reduce `--batch-size` or `--image-size`, or try `--backbone mobilenetv2`.

---

## Evaluation

### Linear Probe

Trains a single Dense layer on frozen features. You can tap features **before** or **after** the projector.

**Using a SavedModel encoder directory (recommended):**
```bash
# CPU is fine; add --device cpu if you want to pin to CPU explicitly
python3 scripts/eval_linear.py \
  --ckpt checkpoints_tf/vicreg_tf.weights.h5 \
  --dataset cifar10 --image-size 32 \
  --epochs 50 --batch-size 512
```

**Using a Keras weights file (.h5) with an internally built backbone:**
```bash
python scripts/eval_linear.py \
  --dataset cifar10 --image-size 32 \
  --epochs 30 --batch-size 256 \
  --ckpt checkpoints_tf/vicreg_tf.weights.h5 \
  --feat-layer pool \
  --bn-adapt-steps 200 \
  --mixed-bf16
```

**Important flags:**
- `--feat-layer {pool, proj}` chooses pooled backbone features or projector outputs
- `--bn-adapt-steps N` runs a short forward-only pass to adapt BN stats to the evaluation distribution
- `--base-lr` and `--wd` control probe optimization (default values are reasonable)

### k-NN Evaluation

Extracts features for train and test, L2-normalizes them, and does a cosine-similarity k-NN vote.

**SavedModel path:**
```bash
python scripts/knn_eval.py \
  --dataset cifar10 --image-size 32 \
  --k 200 --batch-size 512 \
  --encoder-path artifacts/encoder_savedmodel \
  --feat-layer pool \
  --mixed-bf16
```

**Keras weights path:**
```bash
python scripts/knn_eval.py \
  --dataset cifar10 --image-size 32 \
  --k 200 --batch-size 512 \
  --ckpt checkpoints_tf/vicreg_tf.weights.h5 \
  --feat-layer pool \
  --mixed-bf16
```

**Notes:**
- On CPU, large `--k` can be slow; try `--k 20..200` and adjust `--batch-size`.
- If you use `--image-size 224` on CPU with ResNet50, you will likely need `--batch-size 32` or smaller.

---

## Key Implementation Details

### Data & Augmentations
- Two independent augmented views per image: random resized crop, horizontal flip, color jitter, optional grayscale/blur, and standardization.
- CIFAR / STL10 builders return tuples suitable for Keras (`((view1, view2), 0)` for SSL; `(image, label)` for supervised).

### Backbone & Projector
- Default backbone is **ResNet50V2** with `include_top=False` and a global average pooling layer (named e.g. `feat_pool`).
- The projector is an MLP: `[Dense (no bias) → BatchNorm → ReLU] × (L-1)` then a linear output layer of dimension `proj_out`.

### Losses
- **VICRegLoss:** alignment + variance hinge with fixed \(\gamma\) + covariance off-diagonal penalty.  
- **AdaptiveVICRegLoss:** EMA-based $\gamma_t$ (median of EMA stds, clipped to $[\gamma_{\min}, \gamma_{\max}]$); **trace-normalized covariance** with Frobenius penalty to $(1/d)I$.  
- Loss returns total plus logs: `align`, `var`, `cov`, and `gamma_t` when adaptive is enabled.

### Schedules
- Cosine ramp for $\lambda$ and $\nu$ early in training; improves stability and removes brittle warmup tuning.

### BN Adaptation
- After loading a pretrained encoder, a short forward pass on unlabeled data updates BN running stats without touching weights. Improves linear/k-NN performance when eval distribution differs from pretraining.

### Mixed Precision on CPU/GPU
- `--mixed-bf16` turns on mixed precision: **bfloat16** on CPUs that support it and **float16** on GPUs (via TF’s policy). This often speeds up training/inference with minimal accuracy impact. Disable if you see numerical issues.

---

## Recommended Hyperparameters

- **CIFAR-10 (img=32, ResNet50V2):** `--epochs 200`, `--batch-size 512` (reduce on CPU), `--proj-out 8192`, `--proj-layers 3`, `--lambda0 25`, `--mu0 25`, `--nu0 1`, `--adaptive`, `--use-schedules`.  
- **STL-10 (img=96):** `--batch-size 256`, similar weights; increase epochs for stronger features.  
- **Folder dataset (img=224):** consider `--backbone resnet50v2 --batch-size 128` on GPU, or `--backbone mobilenetv2 --batch-size 32` on CPU.

For linear probe on CIFAR-10, start with: `--epochs 30`, `--base-lr 0.2`, `--wd 1e-4`, `--bn-adapt-steps 200`.

---

## Troubleshooting & Gotchas

- **“Could not find CUDA drivers / GPU will not be used.”**  
  Your TensorFlow install is CPU-only or CUDA/cuDNN do not match. The scripts continue on CPU.

- **KerasTensor cannot be used as input to a TF function.**  
  This happens when mixing raw `tf.*` ops with symbolic KerasTensors. The provided scripts wrap such ops in Keras layers; if you modify them, use `keras.layers.Lambda` or custom layers.

- **OOM on CPU (especially img=224 + ResNet50):**  
  Lower `--batch-size` and/or `--image-size`; try `--mixed-bf16`; or switch to `--backbone mobilenetv2`.

- **`.h5` vs SavedModel loading confusion:**  
  The eval scripts accept **either** `--encoder-path` (SavedModel dir) **or** `--ckpt` (Keras `.weights.h5`). They auto-handle supported formats and print a clear error if the path is wrong.

---

## Reproducibility Tips

- Set seeds (`--seed`) but note full determinism is hard due to multi-threading and non-deterministic kernels.  
- Log everything (CLI flags, commit hash, dataset checksums).  
- Use the same augmentations and image sizes for fair comparisons.  
- Run BN adaptation before probing/evaluating, especially if image size changed between pretraining and eval.

---

## Citations

- **VICReg original paper:**  
  Adrien Bardes, Jean Ponce, Yann LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* arXiv:2105.04906. https://arxiv.org/pdf/2105.04906

- **This repository:**

  ```bibtex
  @software{Kabra_Adaptive_VICReg_TF_2025,
    author = {Nishant Kabra},
    title  = {Adaptive VICReg in TensorFlow and Keras},
    year   = {2025}
  }
  ```

