import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix

#Configuration
DATASET_PATH = "dataset" 
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

#Load Dataset
print("--- Loading Br35H Dataset ---")
train_data_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    class_names=['no', 'yes'],
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_data_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    class_names=['no', 'yes'],
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data_raw.map(lambda x, y: (data_augmentation(x, training=True)/255.0, y)).prefetch(AUTOTUNE)
val_data = val_data_raw.map(lambda x, y: (x/255.0, y)).prefetch(AUTOTUNE)


base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    Input(shape=IMAGE_SIZE + (3,)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)), 
    Dropout(0.7),
    Dense(1, activation='sigmoid')
])

# Training
print("\n--- Phase 1: Feature Extraction ---")
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history_p1 = model.fit(train_data, validation_data=val_data, epochs=5)

print("\n--- Phase 2: Fine-Tuning ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
history_p2 = model.fit(train_data, validation_data=val_data, epochs=10)


val_loss, val_acc = model.evaluate(val_data)
print(f"\n Final Validation Accuracy: {val_acc*100:.2f}%")


print("\n--- Generating Training Graphics ---")
acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
val_acc_list = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
loss = history_p1.history['loss'] + history_p2.history['loss']
val_loss_list = history_p1.history['val_loss'] + history_p2.history['val_loss']

plt.figure(figsize=(14, 6))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy', color='teal', linewidth=2)
plt.plot(val_acc_list, label='Val Accuracy', color='orange', linestyle='--')
plt.axvline(x=4, color='red', linestyle=':', label='Fine-tuning Start')
plt.title('Accuracy: Phase 1 & 2')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss', color='teal', linewidth=2)
plt.plot(val_loss_list, label='Val Loss', color='orange', linestyle='--')
plt.axvline(x=4, color='red', linestyle=':', label='Fine-tuning Start')
plt.title('Loss: Phase 1 & 2')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig('results/main_approach_metrics.png')
print("[INFO] Graphics saved to 'results/main_approach_metrics.png'")
plt.show()

print("\n--- Generating Balanced Evaluation Metrics ---")

val_data_final_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='binary',
    class_names=['no', 'yes'],
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True # Shuffling ensures we get a mix of both classes 
)

val_data_final = val_data_final_raw.map(lambda x, y: (x/255.0, y))

y_true = []
y_pred = []

for x, y in val_data_final:
    y_true.extend(y.numpy())
    preds = model.predict(x, verbose=0)
    y_pred.extend((preds > 0.5).astype(int))

y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()

print("\nClassification Report (Balanced Support):")
print(classification_report(y_true, y_pred, target_names=['no', 'yes']))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (Balanced Support):")
print(cm)

#Save
os.makedirs("models", exist_ok=True)
model.save("models/main_approach_mobilenet.keras")
print("\n Final Model saved to 'models/main_approach_mobilenet.keras'")