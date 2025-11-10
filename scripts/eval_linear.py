import os
import math
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
from tensorflow.keras import layers, regularizers

# -------- Runtime knobs via env --------
if os.getenv("MIXED_BF16", "0") == "1":
    mixed_precision.set_global_policy("mixed_bfloat16")
    print("[eval_linear] mixed_bfloat16 enabled")

BACKBONE = os.getenv("BACKBONE", "resnet50v2").lower()
BN_BATCH = int(os.getenv("BN_BATCH", "32"))

# -------- Repro --------
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
set_seed(42)

# -------- Data --------
def load_cifar(dataset: str):
    if dataset.lower() == "cifar100":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100
    elif dataset.lower() == "cifar10":
        (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    ytr = ytr.reshape(-1).astype("int32")
    yte = yte.reshape(-1).astype("int32")
    return (xtr, ytr), (xte, yte), num_classes

def _augment_train(img, label, image_size, num_classes):
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.image.random_flip_left_right(img)
    pad = 4 if image_size <= 64 else max(2, image_size // 64)
    img = tf.image.resize_with_crop_or_pad(img, image_size + pad, image_size + pad)
    img = tf.image.random_crop(img, size=(image_size, image_size, 3))
    label = tf.one_hot(label, num_classes)
    return img, label

def _augment_val(img, label, image_size, num_classes):
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    label = tf.one_hot(label, num_classes)
    return img, label

def make_datasets(dataset, image_size, batch_size):
    (xtr, ytr), (xte, yte), num_classes = load_cifar(dataset)
    xtr = tf.convert_to_tensor(xtr, dtype=tf.uint8)
    xte = tf.convert_to_tensor(xte, dtype=tf.uint8)
    ytr = tf.convert_to_tensor(ytr, dtype=tf.int32)
    yte = tf.convert_to_tensor(yte, dtype=tf.int32)

    ds_train = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True)
    ds_train = ds_train.map(
        lambda a, b: _augment_train(tf.cast(a, tf.float32), b, image_size, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((xte, yte))
    ds_val = ds_val.map(
        lambda a, b: _augment_val(tf.cast(a, tf.float32), b, image_size, num_classes),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_train_images = tf.data.Dataset.from_tensor_slices(xtr)
    ds_train_images = ds_train_images.shuffle(10000, reshuffle_each_iteration=True)
    ds_train_images = ds_train_images.map(
        lambda a: tf.image.resize(tf.cast(a, tf.float32), (image_size, image_size)),
        num_parallel_calls=tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_train_images, num_classes, xtr.shape[0], xte.shape[0]

# -------- Encoder + Feature Extractor --------
def build_encoder(image_size: int):
    if BACKBONE == "resnet50":
        base = keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=(image_size, image_size, 3), pooling="avg"
        )
        preprocess = keras.applications.resnet.preprocess_input
    else:
        base = keras.applications.ResNet50V2(
            include_top=False, weights=None, input_shape=(image_size, image_size, 3), pooling="avg"
        )
        preprocess = keras.applications.resnet_v2.preprocess_input
    enc = keras.Model(base.input, base.output, name=f"encoder_{BACKBONE}")
    return enc, preprocess

def _legacy_h5_load_by_name(model: keras.Model, path: str, skip_mismatch=True):
    try:
        import h5py
        try:
            # Keras 3
            from keras.saving.legacy import hdf5_format
        except Exception:
            # Older Keras
            from keras.saving import hdf5_format
        with h5py.File(path, "r") as f:
            hdf5_format.load_weights_from_hdf5_group_by_name(
                f, model.layers, skip_mismatch=skip_mismatch
            )
        print(f"[eval_linear] loaded legacy H5 by_name from {path} (skip_mismatch={skip_mismatch})")
        return True
    except Exception as e:
        print(f"[eval_linear] legacy H5 by_name load failed: {e}")
        return False

def load_weights_safely(model: keras.Model, ckpt_path: str):
    if not ckpt_path:
        return
    try:
        # Standard path (Keras 3): no by_name arg
        model.load_weights(ckpt_path, skip_mismatch=True)
        print(f"[eval_linear] loaded weights from {ckpt_path} (skip_mismatch=True)")
    except Exception as e1:
        print(f"[eval_linear] standard load failed: {e1}")
        if not _legacy_h5_load_by_name(model, ckpt_path, skip_mismatch=True):
            print("[eval_linear] WARNING: no weights loaded.")

def l2_normalize_layer(name="l2norm"):
    def _norm(z):
        # use keras.ops only
        sq = keras.ops.square(z)
        ssum = keras.ops.sum(sq, axis=-1, keepdims=True)
        denom = keras.ops.maximum(keras.ops.sqrt(ssum), 1e-12)
        return z / denom
    return layers.Lambda(_norm, name=name)

def build_feature_extractor(ckpt_path: str, image_size: int) -> keras.Model:
    enc, preprocess = build_encoder(image_size)
    load_weights_safely(enc, ckpt_path)

    inp = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Lambda(lambda t: t, name="identity")(inp)
    x = layers.Lambda(preprocess, name="preprocess")(x)
    feats = enc(x, training=False)
    feats = l2_normalize_layer()(feats)
    return keras.Model(inp, feats, name="feature_extractor")

# -------- BN Adaptation --------
def rebatch_images(ds_images, new_bsz):
    return ds_images.batch(new_bsz).prefetch(tf.data.AUTOTUNE)

def bn_adapt(feature_extractor: keras.Model, ds_images, steps: int):
    if steps <= 0:
        return
    print(f"[eval_linear] BN adaptation steps: {steps} (batch={BN_BATCH})")
    for l in feature_extractor.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = True
    it = iter(ds_images)
    for _ in range(steps):
        try:
            imgs = next(it)
        except StopIteration:
            it = iter(ds_images)
            imgs = next(it)
        _ = feature_extractor(imgs, training=True)
    for l in feature_extractor.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False

# -------- Linear Classifier --------
def build_linear_probe(feature_extractor: keras.Model, num_classes: int, weight_decay: float):
    feature_extractor.trainable = False
    inp = keras.Input(shape=feature_extractor.input_shape[1:])
    feats = feature_extractor(inp, training=False)
    logits = layers.Dense(
        num_classes,
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        kernel_initializer="zeros",
        name="linear_head",
        dtype="float32",
    )(feats)
    return keras.Model(inp, logits, name="linear_eval")

# -------- Train/Eval --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--bn-adapt-steps", type=int, default=0)
    ap.add_argument("--base-lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()

    print(f"[eval_linear] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
          f"epochs={args.epochs} ckpt={args.ckpt}")

    ds_train, ds_val, ds_train_images, num_classes, ntr, _ = make_datasets(
        args.dataset, args.image_size, args.batch_size
    )

    feature_extractor = build_feature_extractor(args.ckpt, args.image_size)

    if args.bn_adapt_steps > 0:
        ds_bn = rebatch_images(ds_train_images, BN_BATCH)
        bn_adapt(feature_extractor, ds_bn, steps=args.bn_adapt_steps)

    model = build_linear_probe(feature_extractor, num_classes, args.wd)

    steps_per_epoch = math.floor(ntr / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    base_lr_scaled = args.base_lr * (args.batch_size / 256.0)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr_scaled, decay_steps=total_steps
    )
    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True
    )

    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=[es], verbose=2)

    print("Final eval:")
    test_loss, test_acc = model.evaluate(ds_val, verbose=2)
    print(f"accuracy: {test_acc:.4f} - loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
