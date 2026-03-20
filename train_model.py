import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# ==============================
# Configuration
# ==============================
DATASET_PATH = "PlantVillage"   # Folder containing your 38 folders
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50  # We use more epochs because EarlyStopping will stop it at the perfect time

# ==============================
# Load & Prepare Dataset
# ==============================
# Using 'categorical' label_mode for better compatibility with Transfer Learning
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Loaded {num_classes} classes: {class_names}")

# Optimize for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================
# Data Augmentation (The Accuracy Secret)
# ==============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
])

# ==============================
# Build Model (Transfer Learning)
# ==============================
# 1. Base Model: MobileNetV2 (Pre-trained)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze it for the first phase

# 2. Combine into one model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1), # MobileNetV2 needs [-1, 1] range
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),  # Prevents overfitting
    layers.Dense(num_classes, activation='softmax')
])

# ==============================
# Compile & Smart Callbacks
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks: Stop early if it stops improving, and lower LR if it plateaus
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# ==============================
# Phase 1: Training the Head
# ==============================
print("\n🚀 Phase 1: Training the custom layers...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

# ==============================
# Phase 2: Fine-Tuning (The 98% Boost)
# ==============================
print("\n🚀 Phase 2: Fine-Tuning the entire model...")
base_model.trainable = True # Unfreeze the brain

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Tiny LR for precision
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train for 10 more epochs to perfect the weights
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# ==============================
# Save & Plot
# ==============================
model.save("plant_disease_model.keras")
print("\n✅ Final Model saved as plant_disease_model.keras")

# Combine history for plotting
total_acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
total_val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
total_loss = history.history['loss'] + fine_tune_history.history['loss']
total_val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(total_acc, label='Train Accuracy')
plt.plot(total_val_acc, label='Val Accuracy')
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning Starts')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(total_loss, label='Train Loss')
plt.plot(total_val_loss, label='Val Loss')
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--', label='Fine-tuning Starts')
plt.title('Loss')
plt.legend()
plt.show()