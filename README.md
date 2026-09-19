# Adaptive VICReg

A TensorFlow/Keras implementation of VICReg [1] with an experimental variant that adjusts the loss weights during training. It pretrains a small CNN encoder on CIFAR-10 [2] without labels, then evaluates it with a linear probe and a kNN classifier. Final project for CS 584, Fall 2025.

## How it works

VICReg trains an encoder by embedding two augmented views of each image and asking three things of the embeddings:

- **Invariance:** the two views should map to nearby points.
- **Variance:** every embedding dimension should keep some spread across the batch, which prevents collapse.
- **Covariance:** different dimensions should be decorrelated.

The paper weights the three terms 25, 25 and 1. This repo adds two optional mechanisms on top. Both are off by default, and with neither flag you get plain VICReg.

**`--adaptive`** picks the three weights again at every step, using two signals:

1. *Magnitude balancing.* A moving average of each raw loss term sets a multiplier (clipped to 0.2x-5x). Terms that are large compared to the others get smaller weights, so none dominates just because of its scale.
2. *Embedding health.* Moving averages of the embedding's mean per-dimension std and mean squared off-diagonal correlation boost the variance weight when std falls below 1 and the covariance weight when correlation rises (up to 4x).

**`--adaptive-targets`** changes what the loss aims for instead of how the terms are weighted. The correlation target `nu` decays from 1.0 to 0.0 over training, and the variance floor `gamma` stays at 1.0. It can be combined with `--adaptive`.

The encoder is a six-layer CNN (three stages of two 3x3 conv, batch norm, ReLU blocks) with global average pooling and a linear layer. The projector is an MLP with 1 to 3 layers.

## Setup

```bash
git clone https://github.com/nkabra56/adaptive-vicreg.git
cd adaptive-vicreg

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need TensorFlow 2.15 or newer, on a Python version that TensorFlow release supports. The code was developed on TF 2.17 (NVIDIA NGC container), and the tests also pass on TF 2.20 with Python 3.13. A GPU makes pretraining much faster but isn't required.

### Docker

The `Dockerfile` builds on NVIDIA's `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` image (TF 2.17, CUDA 12.8), which is what the GPU runs in [EXPERIMENTS.md](EXPERIMENTS.md) used. The repo is mounted at run time, and mounting `.cache_keras` keeps the CIFAR-10 download between runs:

```bash
docker build -t adaptive-vicreg:ngc .
docker run --gpus all -it --rm \
  -v "$(pwd):/workspace" -v "$(pwd)/.cache_keras:/root/.keras" -w /workspace \
  adaptive-vicreg:ngc bash
```

## Usage

Run everything from the repository root. Each script's `--help` lists all options.

### Pretraining

```bash
python3 scripts/train_vicreg.py \
  --dataset cifar10 --epochs 100 --batch-size 256 \
  --feat-dim 2048 --proj-out 4096 --proj-layers 3 \
  --lr 0.01 --wd 1e-6 --use-schedules \
  --model-dir checkpoints_tf --run-name pretrain-c10_baseline
```

Add `--adaptive` for Adaptive VICReg, `--adaptive-targets` for the target schedule, `--seed N` for a reproducible run, and `--device cpu` to force the CPU. `--warmup-epochs N` (with `--use-schedules`) ramps the LR up before the cosine decay, `--clipnorm X` clips gradients, and `--stop-epoch N` stops early while keeping the LR schedule sized for `--epochs`. The results below used `--batch-size 512`, which ran out of memory on an 8 GB GPU.

Each run writes to `checkpoints_tf/<run-name>_<timestamp>/`:

- `vicreg_full.weights.h5`: trainer weights (encoder, projector, optimizer) at the lowest training loss.
- `vicreg_encoder.weights.h5`: the encoder weights from that same checkpoint. This is the file the evaluation scripts load.
- `vicreg_full_last.weights.h5` and `vicreg_encoder_last.weights.h5`: the state after the final epoch, so the end of a run can be compared with the best-loss checkpoint.
- `train_config.json`: the hyperparameters used.
- `metrics/history.jsonl`: one record per epoch with the loss terms, the loss weights, and embedding statistics.

### Resuming a run

```bash
python3 scripts/resume_pretrain.py \
  --ckpt checkpoints_tf/<run>/vicreg_full.weights.h5 \
  --initial-epoch 80 --epochs 200 --resume-lr 0.003
```

`--initial-epoch` is the epoch the original run stopped at and `--epochs` is the epoch to train up to. With `--use-schedules` the cosine schedule continues from that epoch, and `--warmup-steps N` ramps the LR up over the first N steps after resuming. Pass the same `--feat-dim`, `--proj-out`, `--proj-layers`, `--adaptive` and `--adaptive-targets` values as the original run.

### Linear evaluation

Trains one linear layer on frozen encoder features and appends a row with the test accuracy and hyperparameters to a CSV.

```bash
python3 scripts/eval_linear.py \
  --encoder-ckpt checkpoints_tf/<run>/vicreg_encoder.weights.h5 \
  --epochs 10 --lr 0.003 --l2 1e-4 --opt adamw \
  --out-csv results/<run>/linear_eval.csv --method-name VICReg
```

### kNN evaluation

Classifies each test image by a temperature-weighted vote among its `k` nearest training images (cosine similarity on the encoder features) and appends a row to a CSV.

```bash
python3 scripts/knn_eval.py \
  --encoder-ckpt checkpoints_tf/<run>/vicreg_encoder.weights.h5 \
  --k 200 --temperature 0.1 \
  --out-csv results/<run>/knn_eval.csv --method-name VICReg
```

Both evaluation scripts need the encoder-only weights file, and `--feat-dim` must match pretraining. Use `--method-name` to label the row, for example `VICReg` or `AdaptiveVICReg`.

### Plots and tables

```bash
python3 src/vicreg_tf/report_metrics.py \
  --out-dir reports/compare \
  --history "name=VICReg,path=checkpoints_tf/<baseline-run>/metrics/history.jsonl" \
  --history "name=AdaptiveVICReg,path=checkpoints_tf/<adaptive-run>/metrics/history.jsonl" \
  --linear-csv "name=VICReg,path=results/<baseline-run>/linear_eval.csv" \
  --linear-csv "name=AdaptiveVICReg,path=results/<adaptive-run>/linear_eval.csv" \
  --knn-csv "name=VICReg,path=results/<baseline-run>/knn_eval.csv" \
  --knn-csv "name=AdaptiveVICReg,path=results/<adaptive-run>/knn_eval.csv"
```

This writes loss curves (total, invariance, variance, covariance), embedding statistics, and linear and kNN result tables as CSV and PNG. Each run's `train_config.json` is found next to its metrics folder, and with two or more runs the report also plots kNN accuracy against batch size.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite takes about a minute on a CPU.

## Repository layout

```text
scripts/
  train_vicreg.py           pretraining
  resume_pretrain.py        resume or extend a run
  eval_linear.py            linear probe on a frozen encoder
  knn_eval.py               kNN evaluation on a frozen encoder
  _extract_best_encoder.py  re-save the encoder from a run's best checkpoint
src/vicreg_tf/
  model.py                  encoder, projector, VICRegTrainer
  losses.py                 VICReg loss terms
  schedules.py              cosine schedules, adaptive targets, AdaptiveReweighter
  callbacks.py              LR schedule, warmup, loss guard, metrics logger
  data.py, augment.py       CIFAR tf.data pipelines and augmentations
  utils.py                  seeding, device selection, checkpoint helpers
  report_metrics.py         plots and tables
tests/                      pytest suite
```

## Results

These numbers are from an earlier version in which "Adaptive VICReg" only scheduled the `gamma` and `nu` targets (now `--adaptive-targets`) and did not reweight the loss terms. The reweighting in `--adaptive` is newer and hasn't been benchmarked, so treat the table as history. Both runs used 100 pretraining epochs and batch size 512 on CIFAR-10.

| Run | Linear probe top-1 | kNN top-1 (best over a k/temperature grid) |
|---|---|---|
| Baseline VICReg | 51.95% | 46.39% (k=50, T=0.07) |
| Targets-only variant | 45.08% | 38.37% (k=50, T=0.10) |

The targets-only variant produced slightly more decorrelated features but didn't beat the baseline downstream. More recent baseline runs on a laptop GPU (batch size 256) reach 21.74% linear and 30.35% kNN top-1 and show a loss spike partway through training. Those runs and their analysis are in [EXPERIMENTS.md](EXPERIMENTS.md). Open issues and ideas are in [TASKS.md](TASKS.md).

## License and citation

The code is under a custom non-commercial license (see [LICENSE](LICENSE)). You can view it, clone it, and use it for personal, academic or non-commercial research. Redistribution, publishing modified copies, and commercial use need written permission from the author.

If you build on this work, please cite VICReg and this repository ([citation.cff](citation.cff)).

## References

[1] A. Bardes, J. Ponce and Y. LeCun. "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906).

[2] A. Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." Technical report, University of Toronto, 2009 (CIFAR-10 and CIFAR-100).
