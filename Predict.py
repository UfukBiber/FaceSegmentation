import tensorflow as tf 
import cv2, os
import numpy as np
from Model import GetModel, LoadWeights



def ShowOutput(model):
    isRunning = True
    i = 0
    while isRunning:
        img = cv2.imread(os.path.join("TestImages", "%i.jpg"%i))
        inp = tf.expand_dims(tf.cast(tf.constant(img), tf.float32), 0)
        output = model.predict(inp)[0]
        output = np.where(output >= 0.5, 255, 0).astype(np.uint8)
        cv2.imshow("Mask", output)
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)
        
        if (key == ord("q")):
            isRunning = False
        elif (key == ord("n")):
            i = (i+1)%11
    cv2.destroyAllWindows()




if __name__ == "__main__":
    model = GetModel()
    model = LoadWeights(model)
    ShowOutput(model)







