import tensorflow as tf 
import cv2, os
import numpy as np


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
model.load_weights("Model_1")




def Predict(imgIndex):
    isRunning = True
    i = imgIndex
    while isRunning:
        img = cv2.imread(os.path.join("IMAGES", "%i.jpg"%i))
        inp = tf.expand_dims(tf.cast(tf.constant(img), tf.float32), 0)
        output = model.predict(inp)[0]
        output = np.where(output >= 0.5, 255, 0).astype(np.uint8)
        cv2.imshow("IMAGE", img)
        cv2.imshow("MASK", output)
        key = cv2.waitKey(0)
        if (key == ord("q")):
            isRunning = False
        elif (key == ord("n")):
            i += 1


