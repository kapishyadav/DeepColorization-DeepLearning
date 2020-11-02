import cv2
import glob
import os
import torch

img_dir = "Enter dir"

data = []
for file in glob.glob('face_images/*.jpg'):
	img = cv2.imread(file)
	data.append(img)

#print("Number of images: ", len(data))

