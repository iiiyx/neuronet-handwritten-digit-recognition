from random import randint
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.engine.training import Model

from model import create_model, norm
from train import CHECKPOINT_PATH


def recognize(model: Model, test_imgs, n) -> int:
    img = test_imgs[n]
    x = np.expand_dims(img, axis=0)
    res = model.predict(x)
    print(res[0])
    result = np.argmax(res)
    print(f"recognized: {result}")

    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

    return result


def load_data() -> Tuple[np.array, np.array]:
    _, (test_imgs, test_vals) = mnist.load_data()
    return norm(test_imgs), test_vals


if __name__ == "__main__":
    model = create_model()
    model.load_weights(CHECKPOINT_PATH)

    test_imgs, test_vals = load_data()
    print("\n>>>>Start recognizing\n")
    idx = randint(0, 9999)
    res = recognize(model, test_imgs, idx)
    print("\n<<<<Stop recognizing\n")
    assert res == test_vals[idx]
    # test_tf()
