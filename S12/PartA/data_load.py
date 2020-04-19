# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:40:58 2020

@author: pooja
"""

import torchvision.datasets as datasets
import torch.utils.data as data
import os

def dataload(data_transforms,data_dir1):
    data_dir =data_dir1
    num_workers = {'train_data' : 100,'test_data'   : 0}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train_data', 'test_data']}
    dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=512, shuffle=True, num_workers=num_workers[x])
                  for x in ['train_data', 'test_data']}
    
    return(dataloaders)