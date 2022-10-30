from keras.engine.training import Model
from keras.layers import Dense, Flatten
from keras.models import Sequential


def norm(imgs):
    return imgs / 255


def create_model() -> Model:
    model = Sequential(
        [
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    print(model.summary())

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model
