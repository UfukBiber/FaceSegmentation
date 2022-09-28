import numpy as np
import os 
import matplotlib.pyplot as plt
import cv2
DATA_DIR_IMAGES = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebA-HQ-img"
DATA_DIR_MASKS = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebAMask-HQ-mask-anno"

IMAGES_DIR = "IMAGES"
MASKS_DIR = "MASKS"

MASK_NAMES = ["skin", "cloth",   "neck", "r_eye", "l_eye", "l_lip", "u_lip", "mouth", "nose", "r_brow", "l_brow", 
            "l_ear", "r_ear","hair", "hat", "neck_l", "eye_g", "ear_r"]


MASK_COLORS = dict((name, i) for i, name in enumerate(MASK_NAMES))

COLOR_STEP = 200 // (len(MASK_NAMES)+1)


def ShowImg(img):
    plt.imshow(img)
    plt.show()


def PrepareMasks():
    if not (os.path.exists(MASKS_DIR)):
        os.mkdir(MASKS_DIR)
    maskPaths = {}
    folderPaths = os.listdir(DATA_DIR_MASKS)
    for folder in folderPaths:
        basePath = os.path.join(DATA_DIR_MASKS, folder)
        try:
            filePaths = os.listdir(basePath)
        except:
            continue
        for file in filePaths:
            if not (file[0:5].isdigit()):
                continue
            try:
                maskPaths[int(file[0:5])][file[6:-4]] = os.path.join(basePath, file)
            except:
                maskPaths[int(file[0:5])] = {file[6:-4]:os.path.join(basePath, file)}
    keys = sorted(maskPaths.keys())
    for key in keys:
        mask = np.zeros(shape = (128, 128), dtype = np.uint8)
        for maskName in MASK_NAMES:
            try:
                tempMask = cv2.imread(maskPaths[key][maskName], 0)
                tempMask = cv2.resize(tempMask, (128, 128))
                tempMask = tempMask.astype(np.bool_)
                mask[tempMask] = (MASK_COLORS[maskName])
            except:
                continue
        cv2.imwrite(os.path.join(MASKS_DIR, str(key)+".jpg"), mask)
        print("%i of %i mask are done."%(key, len(keys)), end = "\r")
        if (key == 1000):
            return

            
        
    




if __name__ == "__main__":
    PrepareMasks()