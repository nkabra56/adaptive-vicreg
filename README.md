# Adaptive VICReg in TensorFlow and Keras for CIFAR-10

**Author:** Nishant Kabra  
**Date:** : 11-17-2025
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU  
**Runtime:** NVIDIA TensorFlow NGC container (TensorFlow 2.x) or local Python 3.10+

---

## Overview

This repository contains my clean and minimal TensorFlow and Keras implementation of VICReg style self supervised pretraining on CIFAR-10. I pretrain an encoder plus projector using two independently augmented views of each image, then evaluate frozen features using a linear probe and a k-NN classifier. The training script saves the full trainer weights, a frozen encoder snapshot, the exact run config, and a JSONL history for plots and tables.

**VICReg in one paragraph.** VICReg balances three terms:  
1) **Invariance** pulls embeddings of two views of the same image together.  
2) **Variance** enforces a per feature standard deviation floor to avoid collapse.  
3) **Covariance** penalizes off diagonal correlations to reduce redundancy.  
I discard the projector after pretraining and reuse the encoder for downstream tasks.

---

## Repository layout

```

.
├── requirements.txt
├── scripts/
│   ├── train_vicreg.py          # pretrain encoder + projector, save checkpoints, log metrics
│   ├── eval_linear.py           # linear probe on frozen encoder
│   ├── knn_eval.py              # k-NN evaluation on frozen encoder
│   └── resume_pretrain.py       # resume training from a full checkpoint
├── src/
│   ├── __init__.py
│   └── vicreg_tf/
│       ├── __init__.py          # re-exports builders and utilities
│       ├── augment.py           # CIFAR friendly two view augmentation in tf.data
│       ├── data.py              # CIFAR-10 dataset builders and steps per epoch utility
│       ├── losses.py            # VICReg losses and weights container
│       ├── model.py             # build_encoder, build_projector, trainer class
│       ├── schedules.py         # cosine LR and WD and adaptive targets
│       └── utils.py             # GPU probe, mixed precision, safe load, helpers
├── checkpoints_tf/              # created by training scripts
├── results/                     # created by evaluation scripts
└── reports/                     # created by reporting utilities

````

`.gitignore` excludes `checkpoints_tf`, `results`, and `reports` to keep the repo lean. The scripts create these folders when needed.

---

## Environment

### Option A: NVIDIA TensorFlow NGC container (recommended)

I run everything inside an NVIDIA TensorFlow NGC container so CUDA and cuDNN are matched correctly.

1) Pull a recent TF 2.x image
```bash
docker pull nvcr.io/nvidia/tensorflow:24.08-tf2-py3
````

2. Start an interactive container with GPU access and mount the repo

```bash
PROJECT_DIR="$PWD"   # repo root
docker run --rm -it --gpus all \
  -v "$PROJECT_DIR":"$PROJECT_DIR" -w "$PROJECT_DIR" \
  -e TF_CPP_MIN_LOG_LEVEL=1 \
  nvcr.io/nvidia/tensorflow:24.08-tf2-py3 bash
```

3. Install Python packages inside the container

```bash
python3 -m pip install -U pip
pip install -r requirements.txt
```

The scripts print a small GPU probe at startup so you can confirm kernels run on `/GPU:0`.

### Option B: Local install

Use Python 3.10+ and a TensorFlow 2.x build that matches your NVIDIA driver and CUDA stack. Then run:

```bash
pip install -r requirements.txt
```

If you use `tensorflow-addons`, note that some TFA builds warn about support windows. If you see a warning, you can still proceed or switch to the NGC container.

---

## Dataset

I use **CIFAR-10** only. It downloads automatically from Keras.

* 50,000 training images and 10,000 test images
* 10 classes
* 32 × 32 RGB
* During pretraining I sample two independent views per image using `augment.two_view_map`

No manual dataset setup is required.

---

## Reproducing my runs on CIFAR-10

Below is a full end to end recipe. You can copy and paste blocks as needed.

### 1) Train

Pick a run name. The script writes outputs under `checkpoints_tf/<RUN_NAME>_<timestamp>`.

```bash
# --- 1) Train ---
RUN_NAME="c10_model11_cov1.5"
python3 scripts/train_vicreg.py \
  --dataset cifar10 \
  --image-size 32 \
  --epochs 200 \
  --batch-size 256 \
  --feat-dim 2048 \
  --proj-out 4096 \
  --proj-layers 3 \
  --lr 0.002 \
  --wd 1e-5 \
  --adaptive \
  --use-schedules \
  --metrics-compute-on encoder \
  --metrics-probe-batch 32 \
  --record-every 5 \
  --model-dir checkpoints_tf \
  --run-name "$RUN_NAME" \
```

Artifacts written by the training script:

* `vicreg_full.weights.h5` for encoder plus projector
* `vicreg_encoder.weights.h5` for the frozen encoder used in downstream eval
* `train_config.json` for reproducibility
* `metrics/history.jsonl` with loss parts and simple embedding statistics per epoch

### 2) Resolve the latest run directory and encoder weights

```bash
# --- 2) Pick latest run dir and encoder weights ---
RUN_DIR=$(ls -dt checkpoints_tf/${RUN_NAME}* | head -1)
ENC="${RUN_DIR}/vicreg_encoder.weights.h5"
```

### 3) Ensure result and report directories

```bash
# --- 3) Ensure per-run results and reports dirs ---
mkdir -p "results/${RUN_NAME}" "reports/${RUN_NAME}"
```

### 4) Linear probes

I compare SGD and AdamW because the winner can flip based on projector width and batch size.

```bash
# --- 4) Linear evals (two flavors) ---

# SGD style probe
python3 scripts/eval_linear.py \
  --encoder-ckpt "$ENC" \
  --dataset cifar10 --image-size 32 --batch-size 512 \
  --epochs 100 --lr 0.1 --l2 5e-4 \
  --feat-dim 2048 \
  --opt sgd \
  --out-csv "results/${RUN_NAME}/${RUN_NAME}_linear_sgd.csv" \
  --method-name AdaptiveVICReg

# AdamW style probe
python3 scripts/eval_linear.py \
  --encoder-ckpt "$ENC" \
  --dataset cifar10 --image-size 32 --batch-size 512 \
  --epochs 100 --lr 0.003 --l2 1e-4 \
  --feat-dim 2048 \
  --opt adamw \
  --out-csv "results/${RUN_NAME}/${RUN_NAME}_linear_adam.csv" \
  --method-name AdaptiveVICReg
```

Tips:

* If memory is tight, set `--batch-size 256`.
* If convergence is slow, try `--epochs 120` for the AdamW probe.

### 5) k-NN evaluation

```bash
# --- 5) kNN eval (single baseline) ---
python3 scripts/knn_eval.py \
  --encoder-ckpt "$ENC" \
  --dataset cifar10 --image-size 32 --batch-size 256 \
  --feat-dim 2048 \
  --k 200 --temperature 0.1 \
  --out-csv "results/${RUN_NAME}/${RUN_NAME}_knn_eval.csv" \
  --method-name AdaptiveVICReg
```

### 6) k-NN grid search

This sweep finds a better temperature and K without code changes.

```bash
# --- 6) kNN grid search ---
for k in 50 100 200 400; do
  for t in 0.04 0.07 0.1 0.2 0.5; do
    python3 scripts/knn_eval.py \
      --encoder-ckpt "$ENC" \
      --dataset cifar10 --image-size 32 --batch-size 512 \
      --feat-dim 2048 \
      --k "$k" --temperature "$t" \
      --out-csv "results/${RUN_NAME}/${RUN_NAME}_grid_knn.csv" \
      --method-name AdaptiveVICReg
  done
done
```

### 7) Reporting

The reporting utility collects the training history and the evaluation CSVs and writes plots and summary tables to `reports/<RUN_NAME>/`.

```bash
# --- 7) Report ---
python3 src/vicreg_tf/report_metrics.py \
  --out-dir "reports/${RUN_NAME}" \
  --history "name=AdaptiveVICReg,path=${RUN_DIR}/metrics/history.jsonl" \
  --linear-csv "results/${RUN_NAME}/${RUN_NAME}_linear_adam.csv" \
  --knn-csv    "results/${RUN_NAME}/${RUN_NAME}_knn_eval.csv"
```

### 8) Pick the best rows from CSVs

```bash
# Best linear probe by top1 or acc column
python3 - << 'PY'
import pandas as pd, glob
for path in glob.glob("results/*/*linear_*.csv"):
    try:
        df = pd.read_csv(path)
        metric = [c for c in df.columns if 'top1' in c.lower() or 'acc' in c.lower()][0]
        best = df.sort_values(metric, ascending=False).head(1)
        print(path); print(best.to_string(index=False), "\n")
    except Exception as e:
        print("skip", path, e)
PY

# Inspect top kNN settings
python3 - << 'PY'
import pandas as pd, glob, os
cands = glob.glob(os.path.join("results","*","*_knn_*.csv"))
if cands:
    df = pd.read_csv(cands[0])
    metric = [c for c in df.columns if 'top1' in c.lower() or 'acc' in c.lower()][0]
    print(df.sort_values(metric, ascending=False).head(10).to_string(index=False))
else:
    print("No kNN CSV found")
PY
```

---

## Implementation notes

* **Augmentation:** random crop with padding back to target size, horizontal flip, light color jitter, clipped to `[0, 1]`. It is fast and tf.data friendly.
* **Backbone:** compact Keras CNN with feature width `--feat-dim` (default 2048).
* **Projector:** MLP of `--proj-layers` with output width `--proj-out`. Layers have unique names to avoid graph collisions.
* **Loss:** VICReg with weights `(sim, var, cov)`. In training I often use `(25, 25, 1.5)`.
* **Schedules:** optional cosine schedules for learning rate and weight decay.
* **Optimizer:** AdamW when `tensorflow-addons` is available, else Adam.
* **Logging:** JSONL history with total loss, loss parts, and optional embedding statistics to diagnose collapse and redundancy.

---

## Troubleshooting

* **Duplicate layer name error:** Keras Functional graphs require unique layer names. The projector builder in `model.py` assigns distinct names per instantiation. If you customize the projector, make sure names remain unique.
* **GPU memory OOM:** Reduce `--batch-size` and or `--proj-out`. kNN can also OOM for large batch sizes and large K, so try `--batch-size 256` and a smaller K for the search.
* **TFA support warnings:** If `tensorflow-addons` warns about version windows, you can proceed inside the NGC container or swap to plain Adam in the scripts.

---

## Reproducibility checklist

* Hyperparameters written to `train_config.json`
* Encoder snapshot saved as `vicreg_encoder.weights.h5`
* Training history written as `metrics/history.jsonl`
* Evaluation scripts produce CSVs under `results/`
* Report script collects the exact artifacts used for figures and tables

If you need exact seeds, set them at the top of `train_vicreg.py` with `numpy` and `tensorflow` seed functions.

---

## Citations and references

Primary method:

* Adrien Bardes, Jean Ponce, Yann LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** ICLR 2022.

Dataset:

* Alex Krizhevsky. **Learning Multiple Layers of Features from Tiny Images.** Technical Report, University of Toronto, 2009. (CIFAR-10)

Optimizers and schedules used in this repository:

* Ilya Loshchilov and Frank Hutter. **Decoupled Weight Decay Regularization.** ICLR 2019. (AdamW)
* Ilya Loshchilov and Frank Hutter. **SGDR: Stochastic Gradient Descent with Warm Restarts.** ICLR 2017. (cosine schedule idea)

BibTeX snippets:

```bibtex
@inproceedings{bardes2022vicreg,
  title={VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@techreport{krizhevsky2009learning,
  title={Learning Multiple Layers of Features from Tiny Images},
  author={Krizhevsky, Alex},
  institution={University of Toronto},
  year={2009}
}

@inproceedings{loshchilov2019adamw,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{loshchilov2017sgdr,
  title={SGDR: Stochastic Gradient Descent with Warm Restarts},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

---

This code is my original work for academic use in CS-584. CIFAR-10 remains under its original license. Please open issues or pull requests for suggestions and corrections.

```
::contentReference[oaicite:0]{index=0}
```
