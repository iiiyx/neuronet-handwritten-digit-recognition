import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.engine.training import Model
from keras.utils import to_categorical

from model import create_model, norm

CHECKPOINT_PATH = "./checkpoints/cp"


def show_first_imgs(imgs, n):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imgs[i], cmap=plt.cm.binary)
    plt.show()


def train(model: Model, train_imgs, train_vals_cat) -> None:
    # show_first_imgs(train_imgs, 25)
    print("\n>>>>Start training")
    model.fit(train_imgs, train_vals_cat, batch_size=32, epochs=5, validation_split=0.2)
    print("<<<<Stop training")

    model.save_weights(CHECKPOINT_PATH)


def prepare_data():
    (train_imgs, train_vals), (test_imgs, test_vals) = mnist.load_data()

    train_imgs = norm(train_imgs)
    test_imgs = norm(test_imgs)

    train_vals_cat = to_categorical(train_vals, 10)
    test_vals_cat = to_categorical(test_vals, 10)

    return train_imgs, train_vals_cat, test_imgs, test_vals_cat


def test_tf():
    print("TEST")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


if __name__ == "__main__":
    train_imgs, train_vals_cat, test_imgs, test_vals_cat = prepare_data()

    model = create_model()
    train(model, train_imgs, train_vals_cat)

    print("\n>>>>Start evaluating")
    model.evaluate(test_imgs, test_vals_cat)
    print("<<<<Stop evaluating")

    # test_tf()
