import tensorflow as tf 
import Input
import SimpleModel as sm 

import UnetModel as um

import UnetModel_2 as um_v2

IMAGES_DIR = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebA-HQ-img"
MASK_DIR = r"MASKS"
BATCH_SIZE = 32
BUFFER_SIZE = 8

SIMPLE_MODEL_CALLBACK = tf.keras.callbacks.ModelCheckpoint(sm.Model_Path, save_weights_only=True, save_best_only=True)

UNET_MODEL_CALLBACK = tf.keras.callbacks.ModelCheckpoint(um.Model_Path, save_weights_only=True, save_best_only=True)

UNET_MODEL_V2_CALLBACK = tf.keras.callbacks.ModelCheckpoint(um_v2.Model_Path, save_weights_only=True, save_best_only=True)



if __name__ == "__main__":
    image_paths = Input.GetPaths(IMAGES_DIR)
    mask_paths = Input.GetPaths(MASK_DIR)
    train_ds, val_ds = Input.GetTrainAndValDs(image_paths, mask_paths)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(BUFFER_SIZE)



    # simple_model = sm.GetModel(len(Input.MASK_NAMES)+1)
    # # simple_model.load_weights(sm.Model_Path)
    # simple_model.fit(train_ds, validation_data = val_ds, epochs = 10, callbacks = [SIMPLE_MODEL_CALLBACK])

    # unet_model = um.GetModel(len(Input.MASK_NAMES)+1)
    # # unet_model.load_weights(um.Model_Path)
    # unet_model.fit(train_ds, validation_data=val_ds, epochs = 15, callbacks = [UNET_MODEL_CALLBACK])

    unet_model_v2 = um_v2.GetModel(len(Input.MASK_NAMES)+1, (128, 128, 3), 16)
    unet_model_v2.load_weights(um_v2.Model_Path)
    unet_model_v2.fit(train_ds, validation_data=val_ds, epochs = 15, callbacks = [UNET_MODEL_V2_CALLBACK])