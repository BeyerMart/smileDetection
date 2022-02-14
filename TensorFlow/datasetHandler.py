import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
MAIN_PATH = "dataset/"
CATEGORIES = ["positives", "negatives"]

# Parts of code are taken from: https://keras.io/examples/vision/image_classification_from_scratch/


def import_dataset():
    train_ds = image_dataset_from_directory(
        MAIN_PATH,
        seed=1980,
        validation_split=0.3,
        subset="training",
        image_size=[IMG_HEIGHT, IMG_WIDTH],
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )
    val_ds = image_dataset_from_directory(
        MAIN_PATH,
        seed=1980,
        validation_split=0.3,
        subset="validation",
        image_size=[IMG_HEIGHT, IMG_WIDTH],
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    return train_ds, val_ds


def create_dataset():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(MAIN_PATH, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(
                img_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode="grayscale"
            )
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    return data, labels
