# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:06:02 2020

@author: pooja
"""

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np


class album_train:

    def __init__(self):
        self.trans=A.Compose(
                [
                A.PadIfNeeded(min_height=40, min_width=40),
                A.RandomCrop(32,32),
                A.HorizontalFlip(p=0.5),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_height=8,
						min_width=8, fill_value=(np.array([0.4914, 0.4822, 0.4465]))*255.0, p=0.75),
                A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ToTensor()
                
                ])
      
    def __call__(self,img):
        img=np.array(img)
        img=self.trans(image=img)['image']
        return(img)

class album_test:

    def __init__(self):
        self.trans=A.Compose(
                [A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),ToTensor()])

    def __call__(self,img):
        img=np.array(img)
        img=self.trans(image=img)['image']
        return(img)
    