import tensorflow as tf 
import  os
import numpy as np
import cv2 
from Model_2 import GetModel, LoadWeights
from Input import COLOR_STEP, MASK_NAMES
IMAGES_DIR = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebA-HQ-img"

img = cv2.imread(os.path.join(IMAGES_DIR, str(29000)+".jpg"))
img = cv2.resize(img, (128, 128))
model = GetModel(len(MASK_NAMES))
model = LoadWeights(model)


def PlotImg(img):
    img = np.expand_dims(img, axis = 0)
    output = model(img)
    output = np.argmax(output[0], axis = -1)
    output = output * COLOR_STEP
    output = output.astype(np.uint8)
    isRunning = True
    while isRunning:
        cv2.imshow("IMAGE", output)
        key = cv2.waitKey(0)
        if (key == ord("q")):
            isRunning = False


if __name__ == "__main__":
    model = GetModel(len(MASK_NAMES))
    PlotImg(img)








