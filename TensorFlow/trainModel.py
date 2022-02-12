from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import (
    TrueNegatives,
    FalseNegatives,
    FalsePositives,
    TrueNegatives,
)

import math

import matplotlib.pyplot as plt

from datasetHandler import import_dataset, create_dataset

train_ds, val_ds = import_dataset()

print("ds")
print(train_ds)
print("Val")
print(val_ds)

# Parts of code are taken from: https://keras.io/examples/vision/image_classification_from_scratch/
# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation="relu"))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a fourth convolutional layer
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        TrueNegatives(),
        FalseNegatives(),
        FalsePositives(),
        TrueNegatives(),
    ],
)

MODEL_PATH = "model/checkpoint/"
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(train_ds) // BATCH_SIZE  # Batchsize
VALIDATION_STEPS = len(val_ds) // BATCH_SIZE
STEPS_PER_EPOCH = int(math.ceil((1.0 * len(train_ds)) / BATCH_SIZE))
VALIDATION_PER_EPOCH = int(math.ceil((1.0 * len(val_ds)) / BATCH_SIZE))

checkpoint = ModelCheckpoint(
    MODEL_PATH, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
)

history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=50,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint],
)

model.save("model/")

print(history.history.keys())

# Plot training & validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

# Plot training & validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()
