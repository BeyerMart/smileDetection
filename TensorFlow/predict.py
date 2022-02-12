import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

model = tf.keras.models.load_model("model")


def predictImage(filename):
    img = load_img(filename, color_mode="grayscale", target_size=(64, 64))

    plt.imshow(img)

    Y = img_to_array(img)
    X = np.expand_dims(Y, axis=0)

    predictions = model.predict(X)
    score = predictions[0]

    plt.xlabel(
        "Prediction: %.2f%% smile | %.2f%% frown " % (100 * score, 100 * (1 - score)),
        fontsize=12,
    )
    plt.show()


predictImage("dataset/negatives/337.jpg")
