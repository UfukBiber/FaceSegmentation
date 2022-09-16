import tensorflow as tf 
from Data import Data

inputs = tf.keras.Input(shape=(200, 200, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)

x = tf.keras.layers.Conv2DTranspose(256, 3,  activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)

outputs = tf.keras.layers.Conv2D(1, 3, activation="sigmoid",padding="same")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

model.load_weights("Model\Model_1")

data = Data()

def Inppath2Image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = (200, 200))
    img = tf.cast(img, tf.float32) 
    return img

def tarPath2Image(path):
    tar = tf.io.read_file(path)
    tar = tf.io.decode_png(tar, channels = 1)
    tar = tf.image.resize(tar, size = (200, 200))
    tar = tf.cast(tar, tf.bool)
    return tar


test_ds = tf.data.Dataset.from_tensor_slices((data.testImagePath, data.testMaskPath))
test_ds = test_ds.map(lambda x, y: (Inppath2Image(x), tarPath2Image(y))).batch(4)

model.evaluate(test_ds)

