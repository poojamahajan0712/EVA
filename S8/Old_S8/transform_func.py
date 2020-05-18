# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:21:16 2020

@author: pooja
"""
import torch
import torchvision
import torchvision.transforms as transforms
def transform1():
    trans=transforms.Compose(
    [transforms.RandomHorizontalFlip(),  
     transforms.RandomRotation(10),  
     transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),  
     transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return(trans)
