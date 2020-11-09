# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, LeakyReLU, ReLU, CrossEntropyLoss, ConvTranspose2d, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, UpsamplingNearest2d, BatchNorm2d
from torch.optim import Adam, SGD
from torch import tanh

#Convert grayscale to LAB
class ColorNet(Module):
    def __init__(self):
        super(ColorNet, self).__init__()

        # Defining a 2D convolution layer 128*128
        self.conv1  = Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.batch1 = BatchNorm2d(num_features = 64)
        self.relu1  = LeakyReLU(inplace=True)

        # Defining another 2D convolution layer 64*64
        self.conv2  = Conv2d(64,128, kernel_size=8, stride=2, padding=3)
        self.batch2 = BatchNorm2d(num_features = 128)
        self.relu2  = LeakyReLU(inplace=True)

        # Defining another 2D convolution layer 32*32
        self.conv3  = Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.batch3 = BatchNorm2d(num_features = 256)
        self.relu3  = LeakyReLU(inplace=True)

        # Defining another 2D convolution layer 16*16
        self.conv4  = Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.batch4 = BatchNorm2d(num_features = 512)
        self.relu4  = LeakyReLU(inplace=True)

        # Defining another 2D convolution layer 8*8
        self.conv5  = Conv2d(512, 512, kernel_size=3, stride=3, padding=2)
        self.batch5 = BatchNorm2d(num_features = 512)
        self.relu5  = LeakyReLU(inplace=True)

        # UPSAMPLE

        # 4x4 -> 8x8
        self.upSample1 = ConvTranspose2d(in_channels=512, out_channels=256, kernel_size = 2, stride = 2, padding = 0)
        self.upbatch1  = BatchNorm2d(num_features = 256)
        self.uprelu1   = LeakyReLU(inplace=True)

        #8x8 -> 16x16
        self.upSample2 = ConvTranspose2d(in_channels=256, out_channels=128, kernel_size = 2, stride = 2, padding = 0)
        self.upbatch2  = BatchNorm2d(num_features = 128)
        self.uprelu2   = LeakyReLU(inplace=True)

        #16x16 -> 32 x 32
        self.upSample3 = ConvTranspose2d(in_channels=128, out_channels=64, kernel_size = 2, stride = 2, padding = 0)
        self.upbatch3  = BatchNorm2d(num_features = 64)
        self.uprelu3   = LeakyReLU(inplace=True)

        #32x32 -> 64x64
        self.upSample4 = ConvTranspose2d(in_channels=64, out_channels=32, kernel_size = 2, stride = 2, padding = 0)
        self.upbatch4  = BatchNorm2d(num_features = 32)
        self.uprelu4   = LeakyReLU(inplace=True)

        #64x64 -> 128x128
        self.upSample5 = ConvTranspose2d(in_channels=32, out_channels=2, kernel_size = 2, stride = 2, padding = 0)
        self.upbatch5  = BatchNorm2d(num_features = 2)
        self.uptanh = torch.nn.Tanh()


    # Defining the forward pass
    def forward(self, x):
        # x = self.cnn_layers(x)
        x1 = self.relu1(self.batch1(self.conv1(x)))
        x2 = self.relu2(self.batch2(self.conv2(x1)))
        x3 = self.relu3(self.batch3(self.conv3(x2)))
        x4 = self.relu4(self.batch4(self.conv4(x3)))
        x5 = self.relu5(self.batch5(self.conv5(x4)))
        u1 = self.uprelu1(self.upbatch1(self.upSample1(x5)))
        u2 = self.uprelu2(self.upbatch2(self.upSample2(u1)))
        u3 = self.uprelu3(self.upbatch3(self.upSample3(u2)))
        u4 = self.uprelu4(self.upbatch4(self.upSample4(u3)))
        u5 = self.uptanh(self.upbatch5(self.upSample5(u4)))
        return u5
