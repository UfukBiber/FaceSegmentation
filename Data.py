from tkinter.tix import IMAGE
import cv2 
import os 
import re
import shutil
import tensorflow as tf 

DATA_DIR_IMAGES = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebA-HQ-img"
DATA_DIR_MASKS = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebAMask-HQ-mask-anno"

IMAGES_DIR = "IMAGES"
#MASK_DIRS
MASK = "MASKS"
HAIR_DIR = "HAIR"
SKIN_DIR = "SKIN"
RIGHT_EYEBROW_DIR = "RIGHT_EYEBROW"
LEFT_EYEBROW_DIR = "LEFT_EYEBROW"
LOWER_LIP_DIR = "LOWER_LIP"
UPPER_LIP_DIR = "UPPER_LIP"
MOUTH_DIR = "MOUTH"
NECK_DIR = "NECK"
NOSE_DIR = "NOSE"
LEFT_EYE_DIR = "LEFT_EYE"
RIGHT_EYE_DIR = "RIGHT_EYE"
RIGHT_EAR_DIR = "RIGHT_EAR"
LEFT_EAR_DIR = "LEFT_EAR"

class Data:
    def __init__(self) -> None:
        if not (os.path.exists(IMAGES_DIR) and os.path.exists(MASK)):
            self.PrepareImages()
            self.PrepareMasks()
            self.Concanate()
        assert len(os.listdir(IMAGES_DIR)) == len(os.listdir(MASK)), "Images and Mask does not have same length"
        self.trainImagePaths = self.GetPaths(IMAGES_DIR)[0:5000]
        self.trainMaskPaths = self.GetPaths(MASK)[0:5000]
        self.valImagePaths = self.GetPaths(IMAGES_DIR)[5000:7000]
        self.valMaskPaths = self.GetPaths(MASK)[5000:7000]
        self.testImagePath = self.GetPaths(IMAGES_DIR)[7000:8000]
        self.testMaskPath = self.GetPaths(MASK)[7000:8000]

    def PrepareImages(self):
        print("Resizing and moving IMAGES")
        if not (os.path.exists(IMAGES_DIR)):
            os.mkdir(IMAGES_DIR)
        fileNames = os.listdir(DATA_DIR_IMAGES)
        i = 0
        for file in fileNames:
            filePath = os.path.join(DATA_DIR_IMAGES, file)
            img = cv2.imread(filePath)
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(os.path.join(IMAGES_DIR, file), img)
            print("%i / %i " % (i, len(fileNames)), end = "\r")
            i += 1
        print("\n")

    def PrepareMasks(self):
        print("Concanating all masks and moving them to MASK folder")
        folderNames = os.listdir(DATA_DIR_MASKS)
        i = 0
        for folder in folderNames:
            folderPath = os.path.join(DATA_DIR_MASKS, folder)
            files = os.listdir(folderPath)
            j = 0
            for file in files:
                filePath = os.path.join(folderPath, file)
                if (file[-8:-4] == "hair"):
                    if not (os.path.exists(HAIR_DIR)):
                        os.mkdir(HAIR_DIR)
                    newPath = os.path.join(HAIR_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(HAIR_DIR, file), os.path.join(HAIR_DIR, str(int(file[0:5]))+".png"))
                elif (file[-8:-4] == "skin"):
                    if not (os.path.exists(SKIN_DIR)):
                        os.mkdir(SKIN_DIR)
                    newPath = os.path.join(SKIN_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(SKIN_DIR, file), os.path.join(SKIN_DIR, str(int(file[0:5]))+".png"))
                elif (file[-8:-4] == "neck"):
                    if not (os.path.exists(NECK_DIR)):
                        os.mkdir(NECK_DIR)
                    newPath = os.path.join(NECK_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(NECK_DIR, file), os.path.join(NECK_DIR, str(int(file[0:5]))+".png"))
                elif (file[-8:-4] == "nose"):
                    if not (os.path.exists(NOSE_DIR)):
                        os.mkdir(NOSE_DIR)
                    newPath = os.path.join(NOSE_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(NOSE_DIR, file), os.path.join(NOSE_DIR, str(int(file[0:5]))+".png"))
                elif (file[-9:-4] == "l_ear"):
                    if not (os.path.exists(LEFT_EAR_DIR)):
                        os.mkdir(LEFT_EAR_DIR)
                    newPath = os.path.join(LEFT_EAR_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(LEFT_EAR_DIR, file), os.path.join(LEFT_EAR_DIR, str(int(file[0:5]))+".png"))
                elif (file[-9:-4] == "r_ear"):
                    if not (os.path.exists(RIGHT_EAR_DIR)):
                        os.mkdir(RIGHT_EAR_DIR)
                    newPath = os.path.join(RIGHT_EAR_DIR)
                    shutil.copy(filePath, newPath)
                    os.rename(os.path.join(RIGHT_EAR_DIR, file), os.path.join(RIGHT_EAR_DIR, str(int(file[0:5]))+".png"))
                print("Folder %i / %i   File %i / %i" % (i, len(folderNames), j, len(files)), end = "\r")
                j+=1
            i += 1
            print("\n")
    
    def Concanate(self):
        if not (os.path.exists(MASK)):
            os.mkdir(MASK)
        fileNames = os.listdir(IMAGES_DIR)
        i = 0
        for file in fileNames:
            file = file[:-3] + "png"
            mask = cv2.imread(os.path.join(SKIN_DIR, file))
            if (os.path.exists(os.path.join(HAIR_DIR, file))):
                mask = cv2.add(mask, cv2.imread(os.path.join(HAIR_DIR, file)))
            if (os.path.exists(os.path.join(NECK_DIR, file))):
                mask = cv2.add(mask, cv2.imread(os.path.join(NECK_DIR, file)))
            if (os.path.exists(os.path.join(LEFT_EAR_DIR, file))):
                mask = cv2.add(mask, cv2.imread(os.path.join(LEFT_EAR_DIR, file)))
            if (os.path.exists(os.path.join(RIGHT_EAR_DIR, file))):
                mask = cv2.add(mask, cv2.imread(os.path.join(RIGHT_EAR_DIR, file)))
            if (os.path.exists(os.path.join(NOSE_DIR, file))):
                mask = cv2.add(mask, cv2.imread(os.path.join(NOSE_DIR, file)))
            cv2.imwrite(os.path.join(MASK, file), mask)
            print("%i / %i" %(i, len(fileNames)), end = "\r")
            i += 1
    
    def GetPaths(self, dir):
        filePaths = os.listdir(dir)
        for i in range(len(filePaths)):
            filePaths[i] = os.path.join(dir, filePaths[i])
        return filePaths

    def Resize(self, size):
        imagesPaths = os.listdir(IMAGES_DIR)
        maskPaths = os.listdir(MASK)
        i = 0
        print("Images are being resized")
        for imagePath in imagesPaths:
            imagePath = os.path.join(IMAGES_DIR, imagePath)
            img = cv2.imread(imagePath)
            img = cv2.resize(img, size)
            cv2.imwrite(imagePath, img)
            print("%i / %i" % (i, len(imagesPaths)), end = "\r")
        print("\nMasks are being resized")
        i = 0
        for maskPath in maskPaths:
            maskPath = os.path.join(MASK, maskPath)
            mask = cv2.imread(maskPath)
            mask = cv2.resize(mask, size)
            cv2.imwrite(maskPath, mask)
            print("%i / %i" % (i, len(maskPath)), end = "\r")
        
        




            
            

                    



if __name__ == "__main__":
    data = Data()
    # data.PrepareImages()
    # data.PrepareMasks()
    # data.Concanate()