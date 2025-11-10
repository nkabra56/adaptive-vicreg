"""
Resume or fine-tune VICReg/Adaptive-VICReg from a checkpoint.

Author: Nishant Kabra
Date: 11/8/2025
"""
# --- begin repo-root path bootstrap ---
import sys, pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
# --- end repo-root path bootstrap ---

import argparse
import tensorflow as tf
from src.vicreg_tf.model import build_encoder, VICRegModel
from src.vicreg_tf.losses import AdaptiveVICRegLoss, VICRegLoss
from src.vicreg_tf.data import build_cifar10_selfsup, build_cifar100_selfsup, build_stl10_selfsup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "stl10"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--use-schedules", action="store_true")
    p.add_argument("--backbone", type=str, default="resnet50v2",
                   choices=["resnet50", "resnet50v2", "mobilenetv2"])
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--proj-out", type=int, default=2048)
    p.add_argument("--proj-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()


def main():
    a = parse_args()

    if a.dataset == "cifar10":
        ds = build_cifar10_selfsup(a.image_size, a.batch_size)
    elif a.dataset == "cifar100":
        ds = build_cifar100_selfsup(a.image_size, a.batch_size)
    else:
        ds = build_stl10_selfsup(a.image_size, a.batch_size)

    enc = build_encoder(a.backbone, a.image_size, a.proj_hidden, a.proj_out, a.proj_layers)
    loss_layer = AdaptiveVICRegLoss(name="adaptive_vicreg_loss") if a.adaptive else VICRegLoss(name="vicreg_loss")
    model = VICRegModel(encoder=enc, loss_layer=loss_layer, use_schedules=a.use_schedules, name="vicreg_model")

    opt = tf.keras.optimizers.AdamW(learning_rate=a.lr, weight_decay=a.weight_decay)
    model.compile(optimizer=opt)

    # warm build then load weights
    _ = model(tf.zeros([1, a.image_size, a.image_size, 3]), training=False)
    model.load_weights(a.ckpt)

    model.fit(ds, epochs=a.epochs)


if __name__ == "__main__":
    main()
