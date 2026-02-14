import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import json

# -----------------------------
# PARAMETERS
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
DATASET_PATH = "tyre_dataset"

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# ⚠️ FORCE CLASS ORDER HERE
# 0 → defective
# 1 → good
train_data = train_datagen.flow_from_directory(
    DATASET_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["defective", "good"]
)

test_data = test_datagen.flow_from_directory(
    DATASET_PATH + "/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["defective", "good"]
)

# -----------------------------
# VERIFY LABEL MAPPING
# -----------------------------
print("Class indices:", train_data.class_indices)
# Output will be: {'defective': 0, 'good': 1}

# Save labels (BEST PRACTICE)
with open("labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

# -----------------------------
# CUSTOM CLASSIFIER
# -----------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -----------------------------
# COMPILE MODEL
# -----------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("tyre_quality_model.keras")
print("✅ Model saved as tyre_quality_model.keras")

# -----------------------------
# TRAINING GRAPHS
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss")

plt.show()
