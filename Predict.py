import tensorflow as tf 

import Input
import UnetModel as um
import SimpleModel as sm
import UnetModel_2 as um_v2
import matplotlib.pyplot as plt 



def PlotBatch(ds, sm = None, um = None, um_v2=None):
    fig, ax = plt.subplots(5, 5)
    ds = ds.skip(6)
    for x, y in ds.take(1):
        x = tf.cast(x, tf.uint8)
        y = (tf.cast(y, tf.uint8)+1) * 10
        if sm is not None:
            output_sm = sm.predict(x)
            output_sm = tf.argmax(output_sm, axis = -1)
            output_sm = (tf.cast(output_sm, tf.uint8)+1) * 10
        if um is not None:
            output_um = um.predict(x)
            output_um = tf.argmax(output_um, axis = -1)
            output_um = (tf.cast(output_um, tf.uint8)+1) * 10
        if um_v2 is not None:
            output_um_v2 = um_v2.predict(x)
            output_um_v2 = tf.argmax(output_um_v2, axis = -1)
            output_um_v2 = (tf.cast(output_um_v2, tf.uint8)+1) * 10

        for i in range(5):
            if (i == 0):
                ax[0, i].text(0.3, 0.3, "Real Image")
                ax[1, i].text(0.3, 0.3, "Real Mask")
                ax[2, i].text(0.3, 0.3, "SM_Mask")
                ax[3, i].text(0.3, 0.3, "Unet_Mask")
                ax[4, i].text(0.3, 0.3, "Unet_v2_Mask")
            else:
                ax[0, i].imshow(x[i])
                ax[1, i].imshow(y[i])
                ax[2, i].imshow(output_sm[i])
                ax[3, i].imshow(output_um[i])
                ax[4, i].imshow(output_um_v2[i])
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    image_paths = Input.GetPaths(Input.DATA_DIR_IMAGES)
    mask_paths = Input.GetPaths(Input.MASKS_DIR)
    test_ds = Input.GetTestDs(image_paths, mask_paths)
    test_ds = test_ds.batch(32)
    sm_model = sm.GetModel(len(Input.MASK_NAMES)+1)
    sm_model.load_weights(sm.Model_Path)

    um_model = um.GetModel(len(Input.MASK_NAMES)+1)
    um_model.load_weights(um.Model_Path)

    um_v2_model = um_v2.GetModel(len(Input.MASK_NAMES)+1, (128, 128, 3), 16)
    um_v2_model.load_weights(um_v2.Model_Path)
    PlotBatch(test_ds, sm_model, um_model, um_v2_model)















