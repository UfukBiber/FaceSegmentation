import tensorflow as tf 
import cv2, os  


DATA_DIR_MASKS = r"C:\Users\biber\OneDrive\Desktop\Data\CelebAMask-HQ\CelebAMask-HQ-mask-anno\0"

paths = os.listdir(DATA_DIR_MASKS)


filePath = os.path.join(DATA_DIR_MASKS, paths[0])

img = cv2.imread(filePath)
print(img)
print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray)
print(gray.shape)
