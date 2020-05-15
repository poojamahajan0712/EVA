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
                A.HorizontalFlip(),
                A.Cutout(num_holes=2, max_h_size=4, max_w_size=4,fill_value=0.5*255, p=0.5),
                A.Normalize((0.4,0.4,0.4), (0.2,0.2,0.2)),
                ToTensor()
                
                ])
      
    def __call__(self,img):
        img=np.array(img)
        img=self.trans(image=img)['image']
        return(img)

class album_test:

    def __init__(self):
        self.trans=A.Compose(
                [A.Normalize((0.4,0.4,0.4), (0.2,0.2,0.2)),ToTensor()])

    def __call__(self,img):
        img=np.array(img)
        img=self.trans(image=img)['image']
        return(img)
    