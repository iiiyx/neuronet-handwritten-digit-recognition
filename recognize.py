from random import randint

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.engine.training import Model

from main import CHECKPOINT_PATH, create_model, norm, prepare_data


def recognize(model: Model, test_imgs, n):
    img = test_imgs[n]
    x = np.expand_dims(img, axis=0)
    res = model.predict(x)
    print(res[0])
    print(f"recognized: {np.argmax(res)}")

    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()


def load_data() -> np.array:
    _, (test_imgs, _) = mnist.load_data()
    return norm(test_imgs)


if __name__ == "__main__":
    model = create_model()
    model.load_weights(CHECKPOINT_PATH)

    test_imgs = load_data()
    print("\n>>>>Start recognizing\n")
    recognize(model, test_imgs, randint(0, 9999))
    print("\n<<<<Stop recognizing\n")

    # test_tf()
