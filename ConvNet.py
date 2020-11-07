# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer 128*128
            Conv2d(1, 3, kernel_size=8, stride=2, padding=3),
            ReLU(inplace=True),

            # Defining another 2D convolution layer 64*64
            Conv2d(3, 3, kernel_size=8, stride=2, padding=3),
            ReLU(inplace=True),

            # Defining another 2D convolution layer 32*32
            Conv2d(3, 3, kernel_size=4, stride=2, padding=1),
            ReLU(inplace=True),

            # Defining another 2D convolution layer 16*16
            Conv2d(3, 3, kernel_size=4, stride=2, padding=1),
            ReLU(inplace=True),
            
            # Defining another 2D convolution layer 8*8
            Conv2d(3, 3, kernel_size=3, stride=3, padding=1),
            ReLU(inplace=True),
           
            # Defining another 2D convolution layer 4*4
            Conv2d(3, 3, kernel_size=2, stride=2),
            ReLU(inplace=True),
           
            # Defining another 2D convolution layer 2*2
            Conv2d(3, 2, kernel_size=1, stride=1),
            ReLU(inplace=True),
            
        )



    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        return x
