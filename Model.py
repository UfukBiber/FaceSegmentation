import tensorflow as tf 
from Data import Data, IMAGES_DIR, MASK
import matplotlib.pyplot as plt


SIZE = (200, 200)

def Inppath2Image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = SIZE)
    img = tf.cast(img, tf.float32) 
    return img

def tarPath2Image(path):
    tar = tf.io.read_file(path)
    tar = tf.io.decode_png(tar, channels = 1)
    tar = tf.image.resize(tar, size = SIZE)
    tar = tf.cast(tar, tf.bool)
    return tar


data = Data()
# data.Resize(SIZE)


train_ds = tf.data.Dataset.from_tensor_slices((data.trainImagePaths, data.trainMaskPaths))
train_ds = train_ds.map(lambda x, y:(Inppath2Image(x), tarPath2Image(y)), num_parallel_calls=4)

val_ds = tf.data.Dataset.from_tensor_slices((data.valImagePaths, data.valMaskPaths))
val_ds = val_ds.map(lambda x, y:(Inppath2Image(x), tarPath2Image(y)), num_parallel_calls=4)

# for x, y in train_ds.take(30):
#     print(x.shape)
#     print(y.shape)
#     print(y.dtype)
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.imshow(x)
#     ax2.imshow(y)
#     plt.show()

train_ds = train_ds.batch(16).prefetch(16)
val_ds = val_ds.batch(16).prefetch(16)


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


model.fit(train_ds, validation_data = val_ds, callbacks = [tf.keras.callbacks.ModelCheckpoint("Model\Model_1", save_best_only = True, save_weights_only = True)],  epochs = 30)