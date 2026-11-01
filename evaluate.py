
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Dataset settings
dataset_path = "dataset"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Get class names
class_names = val_ds.class_names
print("Classes:", class_names)

# Load trained model
model = tf.keras.models.load_model("mask_detector.keras")

# Store actual and predicted labels
y_true_list = []
y_pred_list = []

# Iterate only once
for images, labels in val_ds:

    predictions = model(images, training=False)

    predicted_classes = tf.argmax(
        predictions, axis=1
    )

    y_true_list.extend(labels.numpy())
    y_pred_list.extend(predicted_classes.numpy())

# Convert into NumPy arrays
y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

# Accuracy
print("\nAccuracy:", accuracy_score(y_true, y_pred) * 100, "%")

# Classification report
print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues")
plt.title("Face Mask Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
