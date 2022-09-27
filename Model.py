import tensorflow as tf 
import os


BATCH_SIZE = 4
BUFFER_SIZE = 4 
VALIDATION_RATIO = 0.2
EPOCHS = 30

MODEL_SAVE_DIR = "Model\Model"


def GetModel():
    print("Preparing the Model")
    inputs = tf.keras.Input(shape=(128, 128, 3))
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
    return model 

def LoadWeights(model):
    try:
        model.load_weights(MODEL_SAVE_DIR)
        print("\nLoaded the weights from %s"%MODEL_SAVE_DIR)
    except:
        print("\nCould not load the weights\n")
    finally:
        return model



if __name__ == "__main__":
    inpPaths = GetPaths("IMAGES")
    tarPaths = GetPaths("MASKS")

    trainInpPaths, valInpPaths = SplitTrainValidation(inpPaths, VALIDATION_RATIO)
    trainTarPaths, valTarPaths = SplitTrainValidation(tarPaths, VALIDATION_RATIO)

    train_ds = GetDataset(trainInpPaths, trainTarPaths)
    val_ds = GetDataset(valInpPaths, valTarPaths)

    model = GetModel()
    model = LoadWeights(model)

    model.fit(train_ds, validation_data = val_ds, callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_DIR, save_best_only = True, save_weights_only = True)],  epochs = EPOCHS)