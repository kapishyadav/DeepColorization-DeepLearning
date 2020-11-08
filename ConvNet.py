# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch import tanh


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Defining a 2D convolution layer 128*128
        self.conv1 = Conv2d(1, 3, kernel_size=8, stride=2, padding=3)
        self.relu1 = ReLU(inplace=True)

        # Defining another 2D convolution layer 64*64
        self.conv2 = Conv2d(3, 3, kernel_size=8, stride=2, padding=3)
        self.relu2 = ReLU(inplace=True)

        # Defining another 2D convolution layer 32*32
        self.conv3 = Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.relu3 = ReLU(inplace=True)

        # Defining another 2D convolution layer 16*16
        self.conv4 = Conv2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.relu4 = ReLU(inplace=True)

        # Defining another 2D convolution layer 8*8
        self.conv5 = Conv2d(3, 3, kernel_size=3, stride=3, padding=2)
        self.relu5 = ReLU(inplace=True)

        # Defining another 2D convolution layer 4*4
        self.conv6 = Conv2d(3, 3, kernel_size=2, stride=2)
        self.relu6 = ReLU(inplace=True)

        # Defining another 2D convolution layer 2*2
        self.conv7 = Conv2d(3, 2, kernel_size=2, stride=1)
        self.tanh = torch.nn.Tanh()


    # Defining the forward pass
    def forward(self, x):
        # x = self.cnn_layers(x)
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.relu3(self.conv3(x2))
        x4 = self.relu4(self.conv4(x3))
        x5 = self.relu5(self.conv5(x4))
        x6 = self.relu6(self.conv6(x5))
        x7 = self.tanh(self.conv7(x6))
        return x7
