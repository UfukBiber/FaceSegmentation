import random 
import cv2 
import os 
import re
import shutil
import tensorflow as tf 

random.seed(1239041251)

DATA_DIR_IMAGES = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebA-HQ-img"
DATA_DIR_MASKS = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebAMask-HQ-mask-anno"

IMAGES_DIR = "IMAGES"
#MASK_DIRS
MASKS_DIR = "MASKS"
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
HAT_DIR = "HAT"

SIZE = (128, 128)

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

def CreateNecessaryFiles():
    print("Creating NecessaryFolders")
    if not (os.path.exists(IMAGES_DIR)):
        os.mkdir(IMAGES_DIR)
    if not (os.path.exists(MASKS_DIR)):
        os.mkdir(MASKS_DIR)
    if not (os.path.exists(HAIR_DIR)):
        os.mkdir(HAIR_DIR)
    if not (os.path.exists(SKIN_DIR)):
        os.mkdir(SKIN_DIR)
    if not (os.path.exists(NECK_DIR)):
        os.mkdir(NECK_DIR)
    if not (os.path.exists(NOSE_DIR)):
        os.mkdir(NOSE_DIR)
    if not (os.path.exists(HAT_DIR)):
        os.mkdir(HAT_DIR)
    if not (os.path.exists(LEFT_EAR_DIR)):
        os.mkdir(LEFT_EAR_DIR)
    if not (os.path.exists(RIGHT_EAR_DIR)):
        os.mkdir(RIGHT_EAR_DIR)


    

def RemoveUnnecessaryFolders():
    print("\nRemoving Unnecessary Folders\n")
    if (os.path.exists(HAIR_DIR)):
        print("\tRemoving %s"%HAIR_DIR)
        shutil.rmtree(HAIR_DIR)
    

    if (os.path.exists(NECK_DIR)):
        print("\tRemoving %s"%NECK_DIR)
        shutil.rmtree(NECK_DIR)
   

    if (os.path.exists(SKIN_DIR)):
        print("\tRemoving %s"%SKIN_DIR)
        shutil.rmtree(SKIN_DIR)
       
    if (os.path.exists(NOSE_DIR)):
        print("\tRemoving %s"%NOSE_DIR)
        shutil.rmtree(NOSE_DIR)
        

    if (os.path.exists(HAT_DIR)):
        print("\tRemoving %s"%HAT_DIR)
        shutil.rmtree(HAT_DIR)
     

    if (os.path.exists(RIGHT_EAR_DIR)):
        print("\tRemoving %s"%RIGHT_EAR_DIR)
        shutil.rmtree(RIGHT_EAR_DIR)
     
    if (os.path.exists(LEFT_EAR_DIR)):
        print("\tRemoving %s"%LEFT_EAR_DIR)
        shutil.rmtree(LEFT_EAR_DIR)
       



def PrepareImages():
    print("\nResizing the images and saving %s folder\n"%IMAGES_DIR)
    print("\t\r")
    filePaths = os.listdir(DATA_DIR_IMAGES)
    i = 0
    for file in filePaths:
        filePath = os.path.join(DATA_DIR_IMAGES, file)
        img = ReadImg(filePath)
        img = Resize(SIZE, img)
        # img = ConvertToGrayScale(img)
        SaveImg(img, os.path.join(IMAGES_DIR, file))
        print("%i / %i " % (i, len(filePaths)), end = "\r")
        i += 1
    print("\n")

def PrepareMasks():
    print("\nProcessing the masks\n")
    print("\t\r")
    folderNames = os.listdir(DATA_DIR_MASKS)
    i = 0
    for folder in folderNames:
        folderPath = os.path.join(DATA_DIR_MASKS, folder)
        files = os.listdir(folderPath)
        j = 0
        for file in files:
            filePath = os.path.join(folderPath, file)
            if (file[-8:-4] == "hair"):
                newPath = os.path.join(HAIR_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(HAIR_DIR, file), os.path.join(HAIR_DIR, str(int(file[0:5]))+".png"))
            elif (file[-8:-4] == "skin"):
                newPath = os.path.join(SKIN_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(SKIN_DIR, file), os.path.join(SKIN_DIR, str(int(file[0:5]))+".png"))
            elif (file[-8:-4] == "neck"):
                newPath = os.path.join(NECK_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(NECK_DIR, file), os.path.join(NECK_DIR, str(int(file[0:5]))+".png"))
            elif (file[-8:-4] == "nose"):
                newPath = os.path.join(NOSE_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(NOSE_DIR, file), os.path.join(NOSE_DIR, str(int(file[0:5]))+".png"))
            elif (file[-7:-4] == "hat"):
                newPath = os.path.join(HAT_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(HAT_DIR, file), os.path.join(HAT_DIR, str(int(file[0:5]))+".png"))
            elif (file[-9:-4] == "l_ear"):
                newPath = os.path.join(LEFT_EAR_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(LEFT_EAR_DIR, file), os.path.join(LEFT_EAR_DIR, str(int(file[0:5]))+".png"))
            elif (file[-9:-4] == "r_ear"):
                newPath = os.path.join(RIGHT_EAR_DIR)
                shutil.copy(filePath, newPath)
                os.rename(os.path.join(RIGHT_EAR_DIR, file), os.path.join(RIGHT_EAR_DIR, str(int(file[0:5]))+".png"))
            print("Folder %i / %i   File %i / %i" % (i, len(folderNames), j, len(files)), end = "\r")
            j+=1
        i += 1
        print("\n")



def ConcanateMasks():
    print("\nConcanating all masks\n")
    print("\t\r")
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
        if (os.path.exists(os.path.join(HAT_DIR, file))):
            mask = cv2.add(mask, cv2.imread(os.path.join(HAT_DIR, file)))
        mask = Resize(SIZE, mask)
        mask = ConvertToGrayScale(mask)
        SaveImg(mask, os.path.join(MASKS_DIR, file))
        print("%i / %i" %(i, len(fileNames)), end = "\r")
        i += 1

def DataAugmentation(percentage):
    imagePaths = os.listdir(IMAGES_DIR)
    imagesToAugment = random.sample(imagePaths, int(len(imagePaths) * percentage))
    for i in range(len(imagesToAugment)):
        img = cv2.imread(os.path.join(IMAGES_DIR, imagesToAugment[i]))
        mask = cv2.imread(os.path.join(MASKS_DIR, imagesToAugment[i][:-3]+"png"))
        if (i % 4 == 0):
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        elif (i % 4 == 1):
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        elif (i % 4 == 2):
            img = tf.image.rot90(img)
            mask = tf.image.rot90(mask)
        else:
            img = tf.image.rot90(img, 3)
            mask = tf.image.rot90(mask, 3)
        imgPath = os.path.join(IMAGES_DIR, str(len(imagePaths)+i)+".jpg")
        maskPath = os.path.join(MASKS_DIR, str(len(imagePaths)+i)+".png")
        SaveImg(img.numpy(), imgPath)
        SaveImg(mask.numpy(), maskPath)        
        print("%i of %i image are augmented."%(i, len(imagesToAugment)), end = "\r")
    print("")




if __name__ == "__main__":
    # CreateNecessaryFiles()
    # PrepareImages()
    # PrepareMasks()
    # ConcanateMasks()
    RemoveUnnecessaryFolders()
    DataAugmentation(0.1)