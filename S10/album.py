# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:21:16 2020

@author: pooja
"""

import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np


class album_train:

    def __init__(self):
        self.trans=A.Compose(
                [
                A.HorizontalFlip(p=0.5),
                #A.Cutout(num_holes=2, max_h_size=4, max_w_size=4,fill_value=0.5*255, p=0.5),
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_height=4,
						min_width=4, fill_value=(np.array([0.4914, 0.4822, 0.4465]))*255.0, p=0.75),
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
    