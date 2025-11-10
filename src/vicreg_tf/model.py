"""
Encoder + projector builder and Keras training model for VICReg.

This module provides:
  - build_encoder(): backbone with global-average-pool and an MLP projector
  - VICRegModel: a custom tf.keras.Model that implements train_step for
    self-supervised learning with VICReg-style losses.

The model is written for Keras 3. Custom args like `loss_layer` are not passed
to super().__init__, which avoids "Unrecognized keyword arguments" errors.

Author: Nishant Kabra
Date: 11/8/2025
"""
from __future__ import annotations

from typing import Optional, Tuple
import tensorflow as tf


# -------------------------
# Backbone + Projector MLP
# -------------------------

def _build_backbone(name: str, image_size: int) -> tf.keras.Model:
    """
    Create a convolutional backbone with no classification head.

    Args:
        name: one of {"resnet50", "resnet50v2", "mobilenetv2"}
        image_size: input side length (square)

    Returns:
        Keras Model that maps [B, H, W, 3] -> [B, H', W', C]
    """
    input_shape = (image_size, image_size, 3)
    if name == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=input_shape
        )
    elif name == "resnet50v2":
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights=None, input_shape=input_shape
        )
    elif name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights=None, input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return base


def build_encoder(
    backbone: str = "resnet50v2",
    image_size: int = 32,
    proj_hidden: int = 2048,
    proj_out: int = 2048,
    proj_layers: int = 2,
) -> tf.keras.Model:
    """
    Build an encoder that outputs projector features z of size proj_out.

    Architecture:
      input -> Backbone(include_top=False) -> GlobalAveragePooling("feat_pool")
            -> [Dense(no bias)->BatchNorm->ReLU] x (proj_layers - 1)
            -> Dense(no bias, units=proj_out, name="projector_out")

    Returns:
        Keras Model mapping image -> z
        The model contains a named pooling layer "feat_pool" that can be used
        later for linear probing if you load the Keras model.
    """
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    base = _build_backbone(backbone, image_size)
    x = base(inputs, training=False)  # backbone frozen behavior is controlled by train_step
    x = tf.keras.layers.GlobalAveragePooling2D(name="feat_pool")(x)

    # Projector MLP
    for i in range(max(0, proj_layers - 1)):
        x = tf.keras.layers.Dense(proj_hidden, use_bias=False, name=f"proj_dense_{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"proj_bn_{i}")(x)
        x = tf.keras.layers.Activation("relu", name=f"proj_relu_{i}")(x)

    z = tf.keras.layers.Dense(proj_out, use_bias=False, name="projector_out")(x)
    model = tf.keras.Model(inputs=inputs, outputs=z, name="encoder_projector")
    return model


# ---------------
# Training model
# ---------------

class VICRegModel(tf.keras.Model):
    """
    Keras Model wrapper that:
      - holds an `encoder` (backbone + projector)
      - holds a `loss_layer` (VICRegLoss or AdaptiveVICRegLoss)
      - implements train_step on two-view self-supervised batches

    Inputs from the dataset should be of the form:
      ((view1, view2), dummy_label)
    where each view is shaped [B, H, W, 3] float32.
    """
    def __init__(
        self,
        encoder: tf.keras.Model,
        loss_layer: tf.keras.layers.Layer,
        use_schedules: bool = True,
        name: str = "vicreg_model",
        **kwargs,
    ):
        # Do NOT pass custom args to super().__init__
        super().__init__(name=name, **kwargs)
        self.encoder = encoder
        self.loss_layer = loss_layer
        self.use_schedules = bool(use_schedules)

        # Simple scalars for optional scheduling (ramp from 0 -> 1)
        # If you want to wire a cosine schedule, you can update these from a callback.
        self.lambda_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="lambda_scale")
        self.nu_scale = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="nu_scale")

    def train_step(self, data):
        # Unpack ((view1, view2), _)
        (v1, v2), _ = data

        with tf.GradientTape() as tape:
            z1 = self.encoder(v1, training=True)
            z2 = self.encoder(v2, training=True)

            total_loss, logs = self.loss_layer(
                [z1, z2],
                lambda_scale=self.lambda_scale,
                nu_scale=self.nu_scale,
                training=True,
            )

        grads = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        # Keras will log anything returned in this dict
        out_logs = {"loss": total_loss}
        # Make sure values are tensors
        for k, v in logs.items():
            out_logs[k] = tf.convert_to_tensor(v)
        return out_logs

    def call(self, inputs, training: Optional[bool] = None):
        # Pass-through for inference: encode a single view
        return self.encoder(inputs, training=training)
    
    def build(self, input_shape=None):
        """Mark the model as built without touching shapes.

        We create variables during the first forward call elsewhere.
        Keeping this no-op avoids fragile shape parsing.
        """
        try:
            super().build(input_shape)
        except TypeError:
            # Some Keras internals may pass an int; ignore and mark built.
            super().build(None)
