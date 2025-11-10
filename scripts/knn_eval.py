import os
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import mixed_precision
from tensorflow.keras import layers

if os.getenv("MIXED_BF16", "0") == "1":
    mixed_precision.set_global_policy("mixed_bfloat16")
    print("[knn_eval] mixed_bfloat16 enabled")

BACKBONE = os.getenv("BACKBONE", "resnet50v2").lower()

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

def _resize_only(img, label, image_size):
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    return img, label

def make_datasets(dataset, image_size, batch_size):
    (xtr, ytr), (xte, yte), num_classes = load_cifar(dataset)
    xtr = tf.convert_to_tensor(xtr, dtype=tf.uint8)
    xte = tf.convert_to_tensor(xte, dtype=tf.uint8)
    ytr = tf.convert_to_tensor(ytr, dtype=tf.int32)
    yte = tf.convert_to_tensor(yte, dtype=tf.int32)

    ds_train = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    ds_train = ds_train.map(lambda a, b: _resize_only(tf.cast(a, tf.float32), b, image_size),
                            num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((xte, yte))
    ds_test = ds_test.map(lambda a, b: _resize_only(tf.cast(a, tf.float32), b, image_size),
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test, num_classes

# -------- Encoder / Feature Extractor --------
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
            from keras.saving.legacy import hdf5_format
        except Exception:
            from keras.saving import hdf5_format
        with h5py.File(path, "r") as f:
            hdf5_format.load_weights_from_hdf5_group_by_name(
                f, model.layers, skip_mismatch=skip_mismatch
            )
        print(f"[knn_eval] loaded legacy H5 by_name from {path} (skip_mismatch={skip_mismatch})")
        return True
    except Exception as e:
        print(f"[knn_eval] legacy H5 by_name load failed: {e}")
        return False

def load_weights_safely(model: keras.Model, ckpt_path: str):
    if not ckpt_path:
        return
    try:
        model.load_weights(ckpt_path, skip_mismatch=True)
        print(f"[knn_eval] loaded weights from {ckpt_path} (skip_mismatch=True)")
    except Exception as e1:
        print(f"[knn_eval] standard load failed: {e1}")
        if not _legacy_h5_load_by_name(model, ckpt_path, skip_mismatch=True):
            print("[knn_eval] WARNING: no weights loaded.")

def l2_normalize_layer(name="l2norm"):
    def _norm(z):
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

# -------- Feature Extraction --------
def extract_features(model: keras.Model, ds):
    feats = []
    labels = []
    for bx, by in ds:
        f = model(bx, training=False)
        feats.append(tf.cast(f, tf.float32).numpy())
        labels.append(by.numpy())
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0).reshape(-1)
    return feats, labels

# -------- kNN (cosine + temperature) --------
def knn_predict(train_X, train_y, test_X, k=200, T=0.07, num_classes=10, chunk=512):
    n_test = test_X.shape[0]
    preds = np.empty(n_test, dtype=np.int32)

    def _l2n(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return x / n
    train_X = _l2n(train_X)
    test_X = _l2n(test_X)

    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        tb = test_X[start:end]                             # [B, D]
        sim = tb @ train_X.T                               # [B, N_train]
        idx = np.argpartition(sim, -k, axis=1)[:, -k:]     # [B, k]
        top_sim = np.take_along_axis(sim, idx, axis=1)     # [B, k]
        order = np.argsort(-top_sim, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        top_sim = np.take_along_axis(top_sim, order, axis=1)

        weights = np.exp(top_sim / max(1e-8, T))           # [B, k]
        B = end - start
        scores = np.zeros((B, num_classes), dtype=np.float32)
        neigh_labels = train_y[idx]                        # [B, k]
        for j in range(k):
            np.add.at(scores, (np.arange(B), neigh_labels[:, j]), weights[:, j])
        preds[start:end] = scores.argmax(axis=1)

    return preds

# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--T", type=float, default=0.07)
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()

    print(f"[knn_eval] dataset={args.dataset} img={args.image_size} bs={args.batch_size} "
          f"k={args.k} T={args.T} ckpt={args.ckpt}")

    ds_train, ds_test, num_classes = make_datasets(args.dataset, args.image_size, args.batch_size)
    feat_extractor = build_feature_extractor(args.ckpt, args.image_size)

    train_X, train_y = extract_features(feat_extractor, ds_train)
    test_X,  test_y  = extract_features(feat_extractor, ds_test)
    print(f"Embedded: train {train_X.shape[0]} / test {test_X.shape[0]} (dim={train_X.shape[1]})")

    preds = knn_predict(train_X, train_y, test_X, k=args.k, T=args.T, num_classes=num_classes, chunk=args.batch_size)
    acc = (preds == test_y).mean()
    print(f"kNN@{args.k} accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
