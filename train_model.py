from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
BATCH_SIZE = 256
EPOCHS = int(os.environ.get("HANDWRITEAI_EPOCHS", "35"))


def normalize_emnist(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    label = tf.cast(label, tf.int32) - 1
    return image, tf.one_hot(label, 26)


def build_model() -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.RandomRotation(0.08)(inputs)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(26, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="handwriteai_emnist_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    (train_ds, test_ds), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    train_count = info.splits["train"].num_examples
    train_ds = (
        train_ds.map(normalize_emnist, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(min(train_count, 20000))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(normalize_emnist, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3),
    ]
    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    best = float(np.max(history.history["val_accuracy"])) * 100
    print(f"Saved {MODEL_PATH}")
    print(f"Best validation accuracy: {best:.2f}%")


if __name__ == "__main__":
    main()
