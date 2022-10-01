# import numpy as np
from cgi import test
from email.mime import image
import os 
import tensorflow as tf
# import cv2
import random

DATA_DIR_IMAGES = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebA-HQ-img"
DATA_DIR_MASKS = r"/home/ufuk/Desktop/CelebAMask-HQ/CelebAMask-HQ-mask-anno"

MASKS_DIR = "MASKS"

MASK_NAMES = ["skin", "cloth",   "neck", "r_eye", "l_eye", "l_lip", "u_lip", "mouth", "nose", "r_brow", "l_brow", 
            "l_ear", "r_ear","hair", "hat", "neck_l", "eye_g", "ear_r"]


MASK_COLORS = dict((name, i) for i, name in enumerate(MASK_NAMES))

TEST_RATIO = 0.05
VAL_RATIO = 0.1
IMAGE_SIZE = (128, 128)

## Concanates the masks from DATA_DIR_MASK according to MASK_COLOR and save them to MASK_DIR
# def PrepareMasks():
#     if not (os.path.exists(MASKS_DIR)):
#         os.mkdir(MASKS_DIR)
#     maskPaths = {}
#     folderPaths = os.listdir(DATA_DIR_MASKS)
#     for folder in folderPaths:
#         basePath = os.path.join(DATA_DIR_MASKS, folder)
#         try:
#             filePaths = os.listdir(basePath)
#         except:
#             continue
#         for file in filePaths:
#             if not (file[0:5].isdigit()):
#                 continue
#             try:
#                 maskPaths[int(file[0:5])][file[6:-4]] = os.path.join(basePath, file)
#             except:
#                 maskPaths[int(file[0:5])] = {file[6:-4]:os.path.join(basePath, file)}
#     keys = sorted(maskPaths.keys())
#     for key in keys:
#         mask = np.ones(shape = (128, 128), dtype = np.uint8)
#         for maskName in MASK_NAMES:
#             try:
#                 tempMask = cv2.imread(maskPaths[key][maskName], 0)
#                 tempMask = cv2.resize(tempMask, (128, 128))
#                 tempMask = tempMask.astype(np.bool_)
#                 mask[tempMask] = (MASK_COLORS[maskName] + 2)
#             except:
#                 continue
#         cv2.imwrite(os.path.join(MASKS_DIR, str(key)+".png"), mask)
#         print("%i of %i mask are done."%(key, len(keys)), end = "\r")

def GetPaths(dir):
    paths = sorted(os.listdir(dir))
    for i in range(len(paths)):
        paths[i] = os.path.join(dir, paths[i])
    return paths 

       
def PathToImage(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    return img 
    
def PathToMask(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels = 1)
    img = tf.cast(img, tf.uint8)-1
    return img


def GetDataset(image_paths, mask_paths):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(lambda x, y : (PathToImage(x), PathToMask(y)), num_parallel_calls = 4)
    return ds 
        
def GetTrainAndValDs(image_paths, mask_paths):
    train_length = int(len(image_paths) * (1 - VAL_RATIO - TEST_RATIO))
    val_length = int(len(image_paths) * (1 - TEST_RATIO))
    random.Random(12341).shuffle(image_paths)
    random.Random(12341).shuffle(mask_paths)
    train_img_paths = image_paths[:train_length]
    train_mask_paths = mask_paths[:train_length]
    val_img_paths = image_paths[train_length:val_length]
    val_mask_paths = mask_paths[train_length:val_length]
    train_ds = GetDataset(train_img_paths, train_mask_paths)
    val_ds = GetDataset(val_img_paths, val_mask_paths)
    return train_ds, val_ds

def GetTestDs(image_paths, mask_paths):
    startInd = int(len(image_paths) * (1 - TEST_RATIO))
    random.Random(12341).shuffle(image_paths)
    random.Random(12341).shuffle(mask_paths)
    test_image_paths = image_paths[startInd:]
    test_mask_paths = mask_paths[startInd:]
    test_ds = GetDataset(test_image_paths, test_mask_paths)
    return test_ds
 


# if __name__ == "__main__":
#     PrepareMasks()