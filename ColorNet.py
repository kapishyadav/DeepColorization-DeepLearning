# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch import tanh

#Convert grayscale to LAB
class ColorNet(Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        # Defining a 2D convolution layer 128*128
        self.conv1 = Conv2d(1, 3, kernel_size=8, stride=2, padding=3)
        self.relu1 = ReLU(inplace=True)
        self.batch1    = BatchNorm2d(num_features = 3)

        # Defining another 2D convolution layer 64*64
        self.conv2 = Conv2d(3, 3, kernel_size=8, stride=2, padding=3)
        self.relu2 = ReLU(inplace=True)
        self.batch2    = BatchNorm2d(num_features = 3)

        # Defining another 2D convolution layer 32*32
        self.conv3 = Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.relu3 = ReLU(inplace=True)
        self.batch3    = BatchNorm2d(num_features = 3)

        # Defining another 2D convolution layer 16*16
        self.conv4  = Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.relu4  = ReLU(inplace=True)
        self.batch4 = BatchNorm2d(num_features = 3)

        # Defining another 2D convolution layer 8*8
        self.conv5  = Conv2d(3, 3, kernel_size=3, stride=3, padding=2)
        self.relu5  = ReLU(inplace=True)
        self.batch5 = BatchNorm2d(num_features = 3)

        # UPSAMPLE

        # 4x4 -> 8x8
        self.upSample1 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size = 2, stride = 2, padding = 0)
        self.uprelu1   = ReLU(inplace=True)
        self.upbatch1  = BatchNorm2d(num_features = 3)

        #8x8 -> 16x16
        self.upSample2 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size = 2, stride = 2, padding = 0)
        self.uprelu2   = ReLU(inplace=True)
        self.upbatch2  = BatchNorm2d(num_features = 3)

        #16x16 -> 32 x 32
        self.upSample3 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size = 2, stride = 2, padding = 0)
        self.uprelu3   = ReLU(inplace=True)
        self.upbatch3  = BatchNorm2d(num_features = 3)

        #32x32 -> 64x64
        self.upSample4 = ConvTranspose2d(in_channels=3, out_channels=3, kernel_size = 2, stride = 2, padding = 0)
        self.uprelu4   = ReLU(inplace=True)
        self.upbatch4  = BatchNorm2d(num_features = 3)

        #64x64 -> 128x128
        self.upSample5 = ConvTranspose2d(in_channels=3, out_channels=2, kernel_size = 2, stride = 2, padding = 0)
        self.uprelu5   = ReLU(inplace=True)
        self.upbatch5  = BatchNorm2d(num_features = 2)


    # Defining the forward pass
    def forward(self, x):
        # x = self.cnn_layers(x)
        x1 = self.batch1(self.relu1(self.conv1(x)))
        x2 = self.batch2(self.relu2(self.conv2(x1)))
        x3 = self.batch3(self.relu3(self.conv3(x2)))
        x4 = self.batch4(self.relu4(self.conv4(x3)))
        x5 = self.batch5(self.relu5(self.conv5(x4)))
        u1 = self.upbatch1(self.uprelu1(self.upSample1(x5)))
        u2 = self.upbatch2(self.uprelu2(self.upSample2(u1)))
        u3 = self.upbatch3(self.uprelu3(self.upSample3(u2)))
        u4 = self.upbatch4(self.uprelu4(self.upSample4(u3)))
        u5 = self.upbatch5(self.uprelu5(self.upSample5(u4)))
        return u5
