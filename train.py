import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib, json, numpy as np, os

Data_DIR = "data"
IMG_SIZE = (64, 64)
SEED = 1337
BATCH = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    Data_DIR, validation_split=0.2, subset="training", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, color_mode="grayscale"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    Data_DIR, validation_split=0.2, subset="validation", seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, color_mode="grayscale"
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

model.export("keras_export")

import tf2onnx
spec = (tf.TensorSpec((1, IMG_SIZE[0], IMG_SIZE[1], 1), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="gestures.onnx"
)
print("saved gestures.onnx and labels.txt")