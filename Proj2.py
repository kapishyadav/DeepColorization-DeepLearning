import cv2
import glob
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

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



RandomIndices = torch.randperm(NumImages)
data = data[RandomIndices, :, :, :]


# test train split

train = data[:675, :, :, :]
test = data[675:, :, :, :]

#data aug
trainset = torch.empty(NumImages*10, 3, 128, 128)
trainset[:675, :, :, :] = data[:675, :, :, :]
# horizontal_flip

horizontal_transform = transforms.Compose([
	    #transforms.ToPILImage(), 
	    transforms.RandomHorizontalFlip(p=1.0),
	    #transforms.ToTensor()
	])
c = 0
for i in np.arange(675, 4*675, 3):
	
	trainset[i, :, :, :] = horizontal_transform(data[c, :, :, :])
	trainset[i+1, :, :, :] = horizontal_transform(data[c, :, :, :])
	trainset[i+2, :, :, :] = horizontal_transform(data[c, :, :, :])
	c = c + 1

crop_transform = transforms.Compose([
	    #transforms.ToPILImage(), 
	    transforms.RandomResizedCrop(128,scale = (0.5,1.0),ratio = (1,1)),
	    #transforms.ToTensor()
	])

c = 0
for i in np.arange(4*675, 7*675, 3):
	
	trainset[i, :, :, :] = crop_transform(data[c, :, :, :])
	trainset[i+1, :, :, :] = crop_transform(data[c, :, :, :])
	trainset[i+2, :, :, :] = crop_transform(data[c, :, :, :])
	c = c + 1
import pdb; pdb.set_trace()


def scale_transform(image):
	rand = np.random.uniform(low = 0.6, high = 1.0)
	image[:,0,:,:] = image[:,0,:,:]*rand
	image[:,1,:,:] = image[:,1,:,:]*rand
	image[:,2,:,:] = image[:,2,:,:]*rand
	return image

c = 0
for i in np.arange(7*675, 10*675, 3):
	
	trainset[i, :, :, :] = scale_transform(data[c, :, :, :])
	trainset[i+1, :, :, :] = scale_transform(data[c, :, :, :])
	trainset[i+2, :, :, :] = scale_transform(data[c, :, :, :])
	c = c + 1
