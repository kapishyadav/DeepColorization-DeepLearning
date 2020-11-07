import cv2
import glob
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ConvNet import ConvNet

#SET SEED!!!!

torch.set_default_tensor_type('torch.FloatTensor')


def scale_transform(image):
	rand = np.random.uniform(low = 0.6, high = 1.0)
	image[0,:,:] = image[0,:,:]*rand
	image[1,:,:] = image[1,:,:]*rand
	image[2,:,:] = image[2,:,:]*rand
	return image

def show_image(image):
	image = np.uint8(torch.squeeze(image.permute(1, 2, 0)))
	# import pdb; pdb.set_trace()
	switched = [2,1,0]
	image = image[:,:,switched]
	imgplot = plt.imshow(image)
	
	plt.show()
	print("Image displayed")
	





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
NumTrainImages = 675
NumTestImages = 75

train = data[:675, :, :, :]
test = data[675:, :, :, :]

#data aug
trainset = torch.empty(NumTrainImages*10, 3, 128, 128)
trainset[:675, :, :, :] = data[:675, :, :, :]
# horizontal_flip

horizontal_transform = transforms.Compose([
	    #transforms.ToPILImage(), 
	    transforms.RandomHorizontalFlip(p=1.0),
	    #transforms.ToTensor()
	])
c = 0
for i in np.arange(NumTrainImages, 4*NumTrainImages, 3):
	
	trainset[i, :, :, :] = horizontal_transform(data[c, :, :, :])
	trainset[i+1, :, :, :] = horizontal_transform(data[c, :, :, :])
	trainset[i+2, :, :, :] = horizontal_transform(data[c, :, :, :])
	c = c + 1

crop_transform = transforms.Compose([
	    #transforms.ToPILImage(), 
	    transforms.RandomResizedCrop(128,scale = (0.5,1.0),ratio = (1.0,1.0)),
	    #transforms.ToTensor()
	])

c = 0
for i in np.arange(4*NumTrainImages, 7*NumTrainImages, 3):
	
	trainset[i, :, :, :] = crop_transform(data[c, :, :, :])
	trainset[i+1, :, :, :] = crop_transform(data[c, :, :, :])
	trainset[i+2, :, :, :] = crop_transform(data[c, :, :, :])
	c = c + 1





c = 0
for i in np.arange(7*NumTrainImages, 10*NumTrainImages, 3):
	
	trainset[i, :, :, :] = scale_transform(torch.squeeze(data[c, :, :, :]))
	trainset[i+1, :, :, :] = scale_transform(torch.squeeze(data[c, :, :, :]))
	trainset[i+2, :, :, :] = scale_transform(torch.squeeze(data[c, :, :, :]))
	c = c + 1

# show_image(trainset[1,:,:,:])
# show_image(torch.squeeze(trainset[678,:,:,:]))
# show_image(torch.squeeze(trainset[6749,:,:,:]))
# show_image(torch.squeeze(trainset[4700,:,:,:]))

trainset_lab = np.zeros((NumTrainImages*10, 128,128,3))


for i in range(0,NumTrainImages*10):
	# 
	trainset_img = torch.squeeze(trainset[i,:,:,:])
	trainset_img = trainset_img.permute(1, 2, 0)
	# import pdb; pdb.set_trace()
	trainset_lab[i,:,:,:] = cv2.cvtColor(np.float32(trainset_img.numpy()), cv2.COLOR_BGR2LAB)

	# L, a, b = cv2.split(trainset_lab)

# Linear Regressor

def LR_ModelDef():
	ConvModel = ConvNet()

	optimizer = torch.optim.Adam(ConvModel.parameters(), lr=0.07)

	loss = torch.nn.MSELoss()

	if torch.cuda.is_available():
		ConvModel = ConvModel.cuda()
		loss = loss.cuda()

	return ConvModel

def train_LR(trainset_lab, ConvModel, epochs):
	trainset_lab = torch.from_numpy(trainset_lab)
	L_channel = np.zeros((NumTrainImages*10, 1, 128, 128))
	a_channel = np.zeros((NumTrainImages*10, 1, 128, 128))
	b_channel = np.zeros((NumTrainImages*10, 1, 128, 128))

	# if torch.cuda.is_available():
	# 	L_channel = L_channel.cuda()
	# 	a_channel = a_channel.cuda()
	# 	b_channel = b_channel.cuda()
	# 	trainset_lab = trainset_lab.cuda()

	for i in range(0, NumTrainImages*10):
		trainset_lab = torch.squeeze(trainset_lab[i,:,:,:])
		trainset_lab = trainset_lab.permute(2,0,1)
		L_channel[i,:,:,:], a_channel[i,:,:,:], b_channel[i,:,:,:] = cv2.split(np.float32(trainset_lab))

		
		L_channel[i,:,:,:] = L_channel[i,:,:,:]/100.0

	optimizer.zero_grad()

	pred = model(L_channel)
	import pdb; pdb.set_trace()
	return pred


model = LR_ModelDef()
pred = train_LR(trainset_lab, model, 5)




















