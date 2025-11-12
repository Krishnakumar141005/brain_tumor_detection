
"""
train.py - trains a tiny CNN on the synthetic brain tumor dataset

Usage:
    python3 train.py

This will create `model.h5` in the project root.
"""
import tensorflow as tf
import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import os

DATA_DIR = "data"
IMG_SIZE = (128, 128)
BATCH = 8

# Check if data directories exist
train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory '{train_dir}' not found. Please ensure the data directory structure is correct.")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory '{val_dir}' not found. Please ensure the data directory structure is correct.")

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=IMG_SIZE,
                                              color_mode="grayscale",
                                              batch_size=BATCH,
                                              class_mode="binary",
                                              shuffle=True)
val_gen = val_datagen.flow_from_directory(val_dir,
                                          target_size=IMG_SIZE,
                                          color_mode="grayscale",
                                          batch_size=BATCH,
                                          class_mode="binary",
                                          shuffle=False)

# Check if data generators found any images
if train_gen.samples == 0:
    raise ValueError(f"No training images found in '{train_dir}'. Please check your data directory.")
if val_gen.samples == 0:
    raise ValueError(f"No validation images found in '{val_dir}'. Please check your data directory.")

print(f"Found {train_gen.samples} training images and {val_gen.samples} validation images.")

model = models.Sequential([
    layers.Input(shape=IMG_SIZE+(1,)),
    layers.Conv2D(16,3,activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32,3,activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(64,3,activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Train briefly to create a demo model
model.fit(train_gen, epochs=3, validation_data=val_gen)
model.save("model.h5")
print("Saved model.h5")
