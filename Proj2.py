import cv2
import glob
import os
import torch
import numpy as np
from torchvision import transforms

#SET SEED!!!!

torch.set_default_tensor_type('torch.FloatTensor')

img_dir = "Enter dir"

#750 images. 750 x 3 x 128 x 128
NumImages = 750
data = torch.empty(NumImages, 3, 128, 128)
c = 0
for file in glob.glob('face_images/*.jpg'):
	img = cv2.imread(file) #B, G, R
	img = torch.from_numpy(np.asarray(img))
	img = img.permute(2, 0, 1)
	data[c, :, :, :] = img
	c = c + 1

import pdb; pdb.set_trace()

RandomIndices = torch.randperm(NumImages)
data = data[RandomIndices, :, :, :]
#print("Number of images: ", len(data))
