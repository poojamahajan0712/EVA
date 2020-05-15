# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:40:58 2020

@author: pooja
"""
import torch
import torchvision

def dataload(train_transform,test_transorm):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transorm)
    
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    
    testloader = torch.utils.data.DataLoader(testset,**dataloader_args)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return(trainloader,testloader,classes)