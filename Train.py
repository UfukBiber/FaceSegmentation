import tensorflow as tf 
from Input import MASK_NAMES, MASK_COLORS, COLOR_STEP, ShowImg
from Model_2 import GetModel, MODEL_SAVE_DIR, LoadWeights
import os 

SEED = 3123124
IMAGES_DIR = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebA-HQ-img"
MASK_DIR = "MASKS"

IMAGE_SIZE = (128, 128)
VAL_RATIO = 0.2


def GetPaths(dir):
    paths = os.listdir(dir)
    for i in range(len(paths)):
        paths[i] = os.path.join(dir, paths[i])
    return paths 

def ReadInpImg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, dtype = tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.
    return img 

def ReadTarImg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 1)
    img = img / COLOR_STEP
    img = tf.image.convert_image_dtype(img, dtype = tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img 

model = GetModel(len(MASK_NAMES))


inpPaths = GetPaths(IMAGES_DIR)[:10000]
tarPaths = GetPaths(MASK_DIR)[:10000]
train_ds = tf.data.Dataset.from_tensor_slices((inpPaths, tarPaths))
train_ds = train_ds.map(lambda x, y: (ReadInpImg(x), ReadTarImg(y)), num_parallel_calls = 4)

train_ds = train_ds.batch(32)


if __name__ == "__main__":
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    model = LoadWeights(model)
    model.fit(train_ds, epochs = 10, callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_DIR, save_weights_only = True)])
  