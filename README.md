# Adaptive VICReg (TensorFlow and Keras) CS584 Final Project

**Author:** Nishant Kabra  
**Date:** 11/8/2025

This repository provides a clear TensorFlow and Keras implementation of VICReg (Variance Invariance Covariance Regularization) with two practical additions that are useful in classroom and research settings.

1. **Adaptive Variance Targeting (AVT)** replaces the fixed variance floor gamma with a data driven target that comes from an exponential moving average of per dimension standard deviations. The median across dimensions is used for robustness.
2. **Scale Invariant Covariance (SICov)** normalizes the covariance matrix by its trace and penalizes the distance to the scaled identity \((1/d)I\) with squared Frobenius norm. This makes the redundancy term less sensitive to global feature scale.

The implementation also includes cosine schedules that smoothly increase the alignment weight lambda and the decorrelation weight nu at the start of training. The goal is stability and simpler tuning.

The code is written with detailed comments. Each function explains inputs, outputs, and the reason for the design. This document shows what each part does and how to run everything on CIFAR, STL10, or a custom folder dataset.

---

## Table of Contents

1. [Background](#background)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
   - [Windows and WSL](#windows-and-wsl)  
   - [macOS and Linux](#macos-and-linux)  
4. [Datasets](#datasets)  
5. [Quick Start](#quick-start)  
6. [How the Code Works](#how-the-code-works)  
   - [Data and Augmentation](#data-and-augmentation)  
   - [Model and Projector](#model-and-projector)  
   - [Losses](#losses)  
   - [Schedules](#schedules)  
   - [Training Loop](#training-loop)  
   - [Evaluation](#evaluation)  
   - [Diagnostics and Plots](#diagnostics-and-plots)  
7. [Configuration and Hyperparameters](#configuration-and-hyperparameters)  
8. [Reproducibility and Performance Tips](#reproducibility-and-performance-tips)  
9. [Extending the Project](#extending-the-project)  
10. [Troubleshooting](#troubleshooting)  
11. [FAQ](#faq)  
12. [Citation and License](#citation-and-license)  
13. [Acknowledgments](#acknowledgments)

---

## Background

VICReg encourages strong representations using three terms on paired augmented views \(z_1, z_2\).

- **Alignment**: \( \mathcal{L}_{\text{align}} = \frac{1}{B} \sum_i \lVert z_{1,i} - z_{2,i} \rVert_2^2 \).  
- **Variance floor**: a hinge that keeps each feature dimension standard deviation above \( \gamma \).  
- **Covariance decorrelation**: a penalty on off diagonal covariance entries to reduce redundancy.

The project keeps the general setup and adds two simple modifications. AVT adapts the variance target during training. SICov penalizes a scale normalized covariance so the loss is not driven by overall feature amplitude.

---

## Repository Structure

```
ADAPTIVE-VICREG-TF/
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ __init__.py
│  └─ vicreg_tf/
│     ├─ __init__.py
│     ├─ augment.py
│     ├─ data.py
│     ├─ losses.py
│     ├─ model.py
│     ├─ schedules.py
│     └─ utils.py
├─ train_vicreg.py
├─ linear_probe.py
├─ eval_knn.py
├─ plots.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ CITATION.cff
└─ .gitignore
```

Generated directories appear at runtime. Examples include `artifacts`, `checkpoints_tf`, `logs`, and `plots`.

---

## Installation

### Windows and WSL

WSL2 with Ubuntu is recommended for TensorFlow GPU. CPU only also works.

```bash
python -m venv .venv
# Linux or WSL bash
source .venv/bin/activate
# PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU on native Windows, install a TensorFlow build that matches your CUDA and cuDNN versions. If you see the message that TensorFlow was not built with CUDA for your GPU, training will run on CPU.

### macOS and Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Linux GPU, install CUDA and cuDNN that match your TensorFlow version before installing the wheel.

---

## Datasets

The code supports four sources.

1. **CIFAR 10 and CIFAR 100** using `tf.keras.datasets`. No extra setup is needed.  
2. **STL10** using `tensorflow_datasets`. The first run downloads to the TFDS cache. You can set `--tfds-data-dir` to control the cache path.  
3. **Folder dataset** arranged with one subfolder per class:
   ```
   data_root/
     class_1/*.jpg
     class_2/*.jpg
     ...
   ```
   Self supervised pretraining ignores the labels. Supervised evaluation uses them.

Image size suggestions: CIFAR uses 32. STL10 uses 96. Folder datasets often use 224.

---

## Quick Start

### Self supervised pretraining

CIFAR 10:

```bash
python train_vicreg.py --dataset cifar10 --image-size 32 \
  --epochs 100 --batch-size 512 --adaptive --mixed
```

STL10 unlabeled:

```bash
python train_vicreg.py --dataset stl10 --image-size 96 \
  --stl10-split unlabeled --epochs 100 --batch-size 256 \
  --adaptive --mixed --tfds-data-dir ~/.cache/tfds
```

Folder dataset:

```bash
python train_vicreg.py --dataset folder --data-root /path/to/data \
  --image-size 224 --epochs 100 --batch-size 256 --adaptive --mixed
```

Artifacts created:

- `checkpoints_tf/vicreg_tf` best weights by lowest training loss  
- `artifacts/encoder_savedmodel` exported encoder for evaluation  
- `logs/train.csv` per epoch metrics

### Linear probe

```bash
python linear_probe.py --dataset cifar10 --image-size 32 \
  --epochs 30 --encoder-path artifacts/encoder_savedmodel
```

### k NN evaluation

```bash
python eval_knn.py --dataset cifar10 --image-size 32 --k 10 \
  --encoder-path artifacts/encoder_savedmodel
```

### Diagnostics and plots

```bash
python plots.py --dataset cifar10 --image-size 32 --probe pool --split test \
  --encoder-path artifacts/encoder_savedmodel --log-csv logs/train.csv
```

Outputs appear in `plots`. Files include `std_hist.png`, `gamma_curve.png`, `cov_heatmap.png`, and `cov_delta_heatmap.png`.

---

## How the Code Works

### Data and Augmentation

`src/vicreg_tf/augment.py` builds two independent augmented views for each input image. The pipeline applies resize, random resized crop, horizontal flip, color jitter, optional grayscale, optional average pool blur, and standardization with ImageNet mean and std. The function `make_two_views` returns a pair `(view1, view2)` of standardized float tensors.

`src/vicreg_tf/data.py` creates data pipelines. CIFAR uses `tf.keras.datasets`. STL10 uses TFDS. The helper `TwoViewsDataset` converts unlabelled streams into `((view1, view2), 0)` which fits Keras input expectations for training. Supervised builders return `(image, label)` for linear probe and k NN.

### Model and Projector

`src/vicreg_tf/model.py` builds a backbone with `include_top=False` and adds a global average pooling layer named `feat_pool`. A projector MLP follows. The projector has repeated blocks `[Dense no bias -> BatchNorm -> ReLU]` and ends with a linear output layer of size `proj_out`.

### Losses

`src/vicreg_tf/losses.py` has two layers.

- `VICRegLoss` computes alignment, a variance hinge with a fixed `gamma`, and an off diagonal covariance penalty.  
- `AdaptiveVICRegLoss` computes per batch standard deviation for each dimension and updates an EMA container. The median of the EMA vector is clipped to `[gamma_min, gamma_max]` to produce `gamma_t`. The covariance is normalized by its trace and the loss penalizes the distance to `(1/d)I` under Frobenius norm.

Both return `(total_loss, logs)` so training can report `align`, `var`, `cov`, and `gamma_t` when available.

### Schedules

`src/vicreg_tf/schedules.py` provides a cosine scaler from 0 to 1 across the full training budget. The training wrapper multiplies `lambda` and `nu` by this factor when schedules are enabled.

### Training Loop

`src/vicreg_tf/model.py` defines `VICRegModel` which overrides `train_step`. The method encodes both views, computes the loss, and applies gradients. It also updates moving average metrics that are visible in logs.

`train_vicreg.py` prepares the dataset and model, compiles with AdamW, and sets callbacks. The encoder SavedModel is exported at the end of training.

### Evaluation

`linear_probe.py` freezes the encoder and trains a single Dense head on top of pooled features or projector outputs.  
`eval_knn.py` extracts features for train and test, normalizes them to unit length, and computes cosine similarities followed by a majority vote among top k neighbors.

### Diagnostics and Plots

`plots.py` shows three aspects:
1. Distribution of per dimension standard deviations as a histogram.  
2. The time series of `gamma_t` if present in `logs/train.csv`.  
3. A heatmap of the trace normalized covariance and a second heatmap of the deviation from `(1/d)I`.

---

## Configuration and Hyperparameters

Use CLI flags or create a small YAML under `configs`. Important flags are listed below. See `train_vicreg.py --help` for the full set.

- Data: `--dataset`, `--data-root`, `--image-size`, `--batch-size`, `--stl10-split`, `--tfds-data-dir`  
- Training: `--epochs`, `--lr`, `--weight-decay`, `--mixed`, `--seed`  
- Encoder and projector: `--backbone`, `--proj-hidden`, `--proj-out`, `--proj-layers`  
- Loss weights: `--lambda0`, `--mu0`, `--nu0`  
- Adaptive settings: `--adaptive` or `--no-adaptive`, `--ema-beta`, `--gamma-min`, `--gamma-max`, `--use-schedules` or `--no-schedules`

Suggestions: CIFAR uses image size 32 and batch size 512 if memory allows. STL10 uses image size 96. Folder datasets often use image size 224 and batch size 256.

---

## Reproducibility and Performance Tips

- Set seeds with `set_seed`. Exact determinism is not guaranteed unless you force deterministic kernels.  
- Mixed precision improves throughput on recent NVIDIA GPUs. Use `--mixed`.  
- Reduce batch size or image size if you hit out of memory.  
- ResNet50V2 is a good default backbone. MobileNetV2 speeds up experiments.  
- Larger batches help covariance estimation during the early epochs.

---

## Extending the Project

- To add a new backbone, modify `_build_backbone` and include a `feat_pool` global pooling layer.  
- To add a new dataset, follow the pattern used for CIFAR or STL10. Provide both self supervised and supervised builders.  
- To change projector depth, adjust `--proj-layers`.  
- To try alternative schedules or optimizers, create a small helper in `schedules.py` or swap the optimizer in `train_vicreg.py`.

---

## Troubleshooting

- If TensorFlow reports that CUDA kernels are not available, your install is CPU only or your CUDA and cuDNN versions do not match the installed wheel.  
- If you see out of memory errors, lower `--batch-size` or `--image-size` and consider MobileNetV2.  
- If k NN accuracy is near random, confirm that pretraining completed and that the probe taps the correct layer.  
- For slow STL10 downloads, set `--tfds-data-dir` to a local path.

---

## FAQ

**Why standardize with ImageNet mean and std on CIFAR**  
This is a common choice for backbones that target ImageNet like behavior. It helps stabilize optimization even when training from scratch.

**Why choose the median of the EMA vector for `gamma_t`**  
The median is robust to a few dimensions with unusually high or low variance. It tends to produce a steady target across iterations.

**Can the encoder be fine tuned end to end**  
Yes. Load the SavedModel, set `trainable=True`, add a small head, and train with a smaller learning rate.

---

## Citation and License

If this project helps your work, please cite:

```
@software{Kabra_Adaptive_VICReg_TF_2025,
  author = {Nishant Kabra},
  title  = {Adaptive VICReg in TensorFlow and Keras},
  year   = {2025}
}
```

The repository template uses Apache 2.0. You can change the license to fit your needs. Keep a `CITATION.cff` file in the root to expose preferred citation details.

---

## Acknowledgments

- VICReg paper by Bardes, Ponce, and LeCun.  
- TensorFlow and Keras documentation and examples that inspired good practices.  
- Classmates and teaching staff for feedback on clarity and ease of use.
