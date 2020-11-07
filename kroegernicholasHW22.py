#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:00:40 2019

@author: nkroeger
"""

import matplotlib.pyplot as plt 

import matplotlib.image as matImg 

def SwitchColors(FName):
    im = matImg.imread(FName)
    switched = [2,1,0]
    im = im[:,:,switched]
    plt.imshow(im)
    plt.show()
    
#SwitchColors('YellowBilledCardinal.jpg') 