import numpy as np
import cv2 
import os 



DATA_DIR_IMAGES = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebA-HQ-img"
DATA_DIR_MASKS = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebAMask-HQ-mask-anno"

IMAGES_DIR = "IMAGES"
MASKS_DIR = "MASKS"

MASK_NAMES = ["background", "skin", "cloth",   "neck", "r_eye", "l_eye", "l_lip", "u_lip", "mouth", "nose", "r_brow", "l_brow", 
            "l_ear", "r_ear","hair", "hat", "neck_l", "eye_g", "ear_r"]


MASK_COLORS = dict((name, i) for i, name in enumerate(MASK_NAMES))

COLOR_STEP = 256 // len(MASK_NAMES)


def ReadImg(path):
    return cv2.imread(path)

def Resize(SIZE, img):
    img = cv2.resize(img, SIZE)
    return img 

def ConvertToGrayScale(img):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    finally:
        return img 

def ShowImg(img):
    isRunning = True 
    while isRunning:
        cv2.imshow("IMAGE", img)
        key = cv2.waitKey(0)
        if (key == ord("q")):
            isRunning = False

def SaveImg(img, path):
    cv2.imwrite(path, img)




def PrepareMasks():
    maskPaths = {}
    folderPaths = os.listdir(DATA_DIR_MASKS)
    for folder in folderPaths:
        basePath = os.path.join(DATA_DIR_MASKS, folder)
        filePaths = os.listdir(basePath)
        for file in filePaths:
            if not (file[0:5].isdigit()):
                continue
            try:
                maskPaths[int(file[0:5])][file[6:-4]] = os.path.join(basePath, file)
            except:
                maskPaths[int(file[0:5])] = {file[6:-4]:os.path.join(basePath, file)}
    totalImages = len(maskPaths.keys())
    for i in range(totalImages):
        mask = np.zeros(shape = (512, 512), dtype= np.uint8)
        for maskName in MASK_NAMES:
            try:
                imgPath = maskPaths[i][maskName]
                img = cv2.imread(imgPath, 0).astype(np.bool_)
                mask[img] = MASK_COLORS[maskName] * COLOR_STEP  
            except:
                pass    
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        maskDir = os.path.join(MASKS_DIR, str(i)+".jpg")
        cv2.imwrite(maskDir, mask)
        print("%i \ 30000" % i, end = "\r")




if __name__ == "__main__":
    PrepareMasks()