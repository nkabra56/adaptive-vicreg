"""
Entry point to train VICReg / Adaptive VICReg in TensorFlow.

It wires up:
  - dataset builders (two augmented views per image)
  - encoder + projector
  - VICReg or AdaptiveVICReg loss
  - training loop with Keras fit()

The script supports CIFAR-10/100 and STL-10.

Author: Nishant Kabra
Date: 11/8/2025
"""
import argparse
import os
import tensorflow as tf

from src.vicreg_tf.data import (
    build_cifar10_selfsup,
    build_cifar100_selfsup,
    build_stl10_selfsup,
)
from src.vicreg_tf.model import build_encoder, VICRegModel
from src.vicreg_tf.losses import VICRegLoss, AdaptiveVICRegLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "stl10"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--backbone", type=str, default="resnet50v2",
                   choices=["resnet50", "resnet50v2", "mobilenetv2"])
    p.add_argument("--proj-hidden", type=int, default=2048)
    p.add_argument("--proj-out", type=int, default=2048)
    p.add_argument("--proj-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--use-schedules", action="store_true")
    p.add_argument("--outdir", type=str, default="checkpoints_tf")
    return p.parse_args()


def main():
    args = parse_args()
    img_size = args.image_size

    os.makedirs(args.outdir, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, "vicreg_tf.weights.h5")

    # ----------------
    # Dataset (two views)
    # ----------------
    if args.dataset == "cifar10":
        ds_train = build_cifar10_selfsup(img_size, args.batch_size)
    elif args.dataset == "cifar100":
        ds_train = build_cifar100_selfsup(img_size, args.batch_size)
    elif args.dataset == "stl10":
        ds_train = build_stl10_selfsup(img_size, args.batch_size)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # ----------------
    # Model + Loss
    # ----------------
    encoder = build_encoder(
        backbone=args.backbone,
        image_size=img_size,
        proj_hidden=args.proj_hidden,
        proj_out=args.proj_out,
        proj_layers=args.proj_layers,
    )

    if args.adaptive:
        loss_layer = AdaptiveVICRegLoss(
            lambda0=25.0, mu0=25.0, nu0=1.0,
            ema_beta=0.99, gamma_min=0.05, gamma_max=1.0,
            name="adaptive_vicreg_loss",
        )
    else:
        loss_layer = VICRegLoss(
            lambda0=25.0, mu0=25.0, nu0=1.0, gamma=1.0, name="vicreg_loss"
        )

    model = VICRegModel(
        encoder=encoder,
        loss_layer=loss_layer,
        use_schedules=args.use_schedules,
        name="vicreg_model",
    )

    # Optimizer and compile
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )
    model.compile(optimizer=optimizer, run_eagerly=False)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # IMPORTANT: Warm-up call so the subclassed model is "built" for Keras 3.
    _ = model(tf.zeros([1, img_size, img_size, 3]), training=False)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Callbacks
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,  # Keras 3 requires built model for weights saving
        monitor="loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    # Train
    model.fit(
        ds_train,
        epochs=args.epochs,
        callbacks=[ckpt_cb],
        verbose=1,
    )


if __name__ == "__main__":
    main()
