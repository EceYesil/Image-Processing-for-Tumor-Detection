import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


# 1. Dataset Path

DATASET_PATH = "dataset"


# 2. Load Dataset

train_data_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

val_data_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary'
)

# Save class names
class_names = train_data_raw.class_names
class_indices = {name: idx for idx, name in enumerate(class_names)}
print("\nClass indices:", class_indices)


# 3. Data Augmentation and Normalization

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data_raw.map(lambda x, y: (data_augmentation(x, training=True)/255.0, y))
train_data = train_data.prefetch(buffer_size=AUTOTUNE)

val_data = val_data_raw.map(lambda x, y: (x/255.0, y))
val_data = val_data.prefetch(buffer_size=AUTOTUNE)


# 4. Build CNN Model

model = Sequential([
    Input(shape=(128,128,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 5. Train Model

EPOCHS = 15

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)


# 6. Evaluate Model

val_loss, val_acc = model.evaluate(val_data)
print(f"\n Validation Accuracy: {val_acc*100:.2f}%")


# 7. Plot Accuracy & Loss

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.savefig('training_results.png')
plt.show()


# 8. Generate Predictions & Report

y_true = np.concatenate([y.numpy() for x, y in val_data], axis=0)
y_pred = np.concatenate([model.predict(x) for x, y in val_data], axis=0)
y_pred_classes = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true, y_pred_classes)
print("\nConfusion Matrix:\n", cm)


# 9. Save the Model 

os.makedirs("models", exist_ok=True)
model.save("models/baseline_cnn.keras")
print("\n Model saved successfully to 'models/baseline_cnn.keras'")
