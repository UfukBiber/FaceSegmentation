import tensorflow as tf 
import cv2, os
import numpy as np
from Model_2 import GetModel, LoadWeights
from Input import COLOR_STEP, MASK_NAMES
IMAGES_DIR = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebA-HQ-img"

def ShowOutput(model):
    isRunning = True
    i = 29000
    while isRunning:
        path = os.path.join(IMAGES_DIR, "%i.jpg"%i)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.
        img = tf.expand_dims(img, axis = 0)
        output = model(img)[0]
        output = np.argmax(output, axis = -1)
        output *= COLOR_STEP
        cv2.imshow("IMAGE", output)
        key = cv2.waitKey(0)
        if (key == ord("q")):
            isRunning = False
        elif (key == ord("n")):
            i = (i+1)%11
    cv2.destroyAllWindows()




if __name__ == "__main__":
    model = GetModel(len(MASK_NAMES))
    model = LoadWeights(model)
    ShowOutput(model)







