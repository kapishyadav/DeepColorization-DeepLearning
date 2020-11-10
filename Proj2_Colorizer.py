import cv2
import glob
import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from skimage.color import rgb2lab, lab2rgb
from skimage import io
from skimage import data
# from skimage.viewer import ImageViewer

from ColorNet import ColorNet

#Boolean, true if you have a trained model to load.
loadModel = True
ModelPath = 'Model/ColorNetHuge.pt'

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'


torch.manual_seed(0)
# torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_tensor_type('torch.DoubleTensor')

def scale_transform(image):
	rand = np.random.uniform(low = 0.6, high = 1.0)
	image[0,:,:] = image[0,:,:]*rand
	image[1,:,:] = image[1,:,:]*rand
	image[2,:,:] = image[2,:,:]*rand
	return image

def show_image(image):
	image = np.uint8(torch.squeeze(image.permute(1, 2, 0)))
	# import pdb; pdb.set_trace()
	imgplot = plt.imshow(image)
	plt.show()
	print("Image displayed")


#750 images. 750 x 3 x 128 x 128
NumImages = 750
data = torch.empty(NumImages, 3, 128, 128)
c = 0
for file in glob.glob('face_images/*.jpg'):
	img = io.imread(file) #R G B
	#import pdb; pdb.set_trace()
	img = torch.from_numpy(np.asarray(img))
	img = img.permute(2, 0, 1) # C *H * W

	data[c, :, :, :] = img
	c = c + 1

RandomIndices = torch.randperm(NumImages)
data = data[RandomIndices, :, :, :]


### test train split ###
NumTrainImages = 675
NumTestImages = 75

train = data[:675, :, :, :]
test = data[675:, :, :, :]

### data aug ###
trainset = torch.empty(NumTrainImages*10, 3, 128, 128)
trainset[:675, :, :, :] = train
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

### Convert from RGB to LAB for TRAIN SET ###

trainset_LAB = np.zeros((NumTrainImages*10, 128,128,3))
for i in range(0,NumTrainImages*10):
	#import pdb; pdb.set_trace()
	trainset_img = torch.squeeze(trainset[i,:,:,:])
	trainset_img = trainset_img.permute(1, 2, 0)
	trainset_LAB[i,:,:,:] = rgb2lab(trainset_img/255.0)

### Convert from RGB to LAB for TEST SET ###
testset_LAB = np.zeros((NumTestImages, 128,128,3))

for i in range(NumTestImages):
	testset_img = torch.squeeze(test[i,:,:,:]) # 128*128*3
	testset_img = testset_img.permute(1, 2, 0)
	testset_LAB[i,:,:,:] = rgb2lab(testset_img/255.0)

### Convert LAB to data & labels FOR TRAIN SET ###
trainset_LAB          = torch.from_numpy(trainset_LAB)
trainset_L_channel    = np.zeros((NumTrainImages*10, 1, 128, 128))
trainset_a_b_channels = np.zeros((NumTrainImages*10, 2, 128, 128))
a_channel = np.zeros((NumTrainImages*10, 1, 128, 128))
b_channel = np.zeros((NumTrainImages*10, 1, 128, 128))
for i in range(0, NumTrainImages*10):
	# import pdb; pdb.set_trace()
	temp_squeezed_img = torch.squeeze(trainset_LAB[i,:,:,:]) #[1,128,128,3] -> [128,128,3]

	trainset_L_channel[i,:,:,:] = temp_squeezed_img[:,:,0]
	a_channel[i,:,:,:] = temp_squeezed_img[:,:,1]
	b_channel[i,:,:,:] = temp_squeezed_img[:,:,2]

	#import pdb; pdb.set_trace()
	#Normalize L channel from [0,100] -> [0,1]
	trainset_L_channel[i,:,:,:] = trainset_L_channel[i,:,:,:]/100.0
	#Normalize a channel from [-110,110] -> [-1,1]
	trainset_a_b_channels[i,0,:,:] = (2.0*(a_channel[i,:,:,:] + 127.0)/254.0) - 1.0
	#Normalize b channel from [-110,110] -> [-1,1]
	trainset_a_b_channels[i,1,:,:] = (2.0*(b_channel[i,:,:,:] + 127.0)/254.0) - 1.0
#Convert to torch
trainset_L_channel    = torch.from_numpy(trainset_L_channel)
trainset_a_b_channels = torch.from_numpy(trainset_a_b_channels)


### Convert LAB to data & labels FOR TEST SET ###
testset_LAB          = torch.from_numpy(testset_LAB)
testset_L_channel    = np.zeros((NumTestImages, 1, 128, 128))
testset_a_b_channels = np.zeros((NumTestImages, 2, 128, 128))
a_channel = np.zeros((NumTestImages, 1, 128, 128))
b_channel = np.zeros((NumTestImages, 1, 128, 128))
for i in range(0, NumTestImages):
	temp_squeezed_img = torch.squeeze(testset_LAB[i,:,:,:]) #[1,128,128,3] -> [128,128,3]

	testset_L_channel[i,:,:,:] = temp_squeezed_img[:,:,0]
	a_channel[i,:,:,:] = temp_squeezed_img[:,:,1]
	b_channel[i,:,:,:] = temp_squeezed_img[:,:,2]

	#Normalize L channel from [0,100] -> [0,1]
	testset_L_channel[i,:,:,:] = testset_L_channel[i,:,:,:]/100.0
	#Normalize a channel from [-110,110] -> [-1,1]
	testset_a_b_channels[i,0,:,:] = (2.0*(a_channel[i,:,:,:] + 127.0)/254.0) - 1.0
	#Normalize b channel from [-110,110] -> [-1,1]
	testset_a_b_channels[i,1,:,:] = (2.0*(b_channel[i,:,:,:] + 127.0)/254.0) - 1.0
#Convert to torch
testset_L_channel    = torch.from_numpy(testset_L_channel)
testset_a_b_channels = torch.from_numpy(testset_a_b_channels)


### TRAINING ###
# Colorizer
def train_Colorizer(ColorModel, optimizer, loss, L_channel, a_b_channels):
	tr_loss = 0
	if torch.cuda.is_available():
		L_channel = L_channel.cuda()
		a_b_channels = a_b_channels.cuda()

	optimizer.zero_grad()
	pred = ColorModel(L_channel)
	# import pdb; pdb.set_trace()
	predError = loss(pred, a_b_channels)
	predError.backward()
	optimizer.step()

	tr_loss = predError.item()
	train_loss_batch.append(tr_loss)

ColorModel = ColorNet()
optimizer  = torch.optim.Adam(ColorModel.parameters(), lr=0.01)
loss       = torch.nn.MSELoss()

if torch.cuda.is_available():
	ColorModel = ColorModel.cuda()
	loss = loss.cuda()


train_loss_batch = []
epochs  = 15
BatchSize = 10
#Train over several epochs
if loadModel==False:
	for i in range(0, epochs):
		train_loss_epoch= []
		print("epoch: ", i+1 )
		#Get batches of size: BatchSize
		for i in np.arange(0, NumTrainImages*10, BatchSize):
			L_channel_batch    = trainset_L_channel[i:(i+10), :, :, :]
			a_b_channels_batch = trainset_a_b_channels[i:(i+10), :, :, :]
			train_Colorizer(ColorModel, optimizer, loss, L_channel_batch, a_b_channels_batch)
		train_loss_epoch.append(np.mean(train_loss_batch))
		print("Train loss: ", np.mean(train_loss_batch))


	#Plot training error
	plt.plot(train_loss_epoch)
	plt.ylabel("Train error")
	plt.xlabel("Epochs")
	plt.savefig("Colorizer_Loss.png")

	###
	#Save the model
	torch.save(ColorModel, ModelPath)

#Load model
if loadModel == True:
	print("Loading model ...")
	ColorModel = torch.load(ModelPath, map_location=map_location)

###
#Prediction on test set
ColorModel.eval() #Since we have batchnorm layers
with torch.no_grad(): #Don't update the weights
	if torch.cuda.is_available():
		testset_L_channel = testset_L_channel.cuda()
		testset_a_b_channels = testset_a_b_channels.cuda()
	pred = ColorModel(testset_L_channel)

# test set loss
testSetLoss = loss(pred, testset_a_b_channels)
print("Test Set Loss: " , testSetLoss.item())

###
#Visualize images
# Merge LAB channels, convert to RGB and visualize
testset_L_channel = testset_L_channel.cpu()
pred = pred.cpu()
testset_a_b_channels = testset_a_b_channels.cpu()

test_RGB = np.zeros((NumTestImages,128,128,3))
for i in range(5):
	#un-normalize L channel from [0,1] -> [0,100]
	L_channel_squeeze = testset_L_channel[i,:,:,:]*100.0
	#un-normalize a and b channel from [-1,1] -> [-110,110]
	a_channel = torch.unsqueeze(pred[i,0,:,:], 0)*127.0
	b_channel = torch.unsqueeze(pred[i,1,:,:], 0)*127.0

	test_merge = np.stack((L_channel_squeeze.numpy(),
		a_channel.numpy(),
		b_channel.numpy()), axis = 0)

	test_merge_transposed = np.transpose(test_merge[:,0,:,:], (1,2,0)) #(3,1,128,128) -> (128,128,3)

	test_RGB[i,:,:,:] = lab2rgb(test_merge_transposed)
	
	plt.imshow(test_RGB[i,:,:,:])
	plt.savefig("Results/fig_{0}.png".format(i))
	#import pdb; pdb.set_trace()




	# test_RGB[i,:,:,:] = test_BGR.permute(2, 1, 0

# for i range(NumTestImages):
# 	plt.savefig(""test_RGB[i,:,:,:])
