# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:19:28 2020

@author: pooja
"""
#import io
import glob
import os
from shutil import move
#from os.path import join
#from os import listdir, rmdir
import random

def sorting():
    target_folder = 'IMagenet/tiny-imagenet-200/val/'
    test_folder   = 'IMagenet/tiny-imagenet-200/train/'
    
    #os.mkdir(test_folder)
    val_dict = {}
    with open('IMagenet/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]
            
    paths = glob.glob('IMagenet/tiny-imagenet-200/val/images/*')
           
    for path in paths:
        file = path.split('/')[-1].split('\\')[-1]
        folder = val_dict[file]
        dest = test_folder + str(folder) + '/images/' + str(file)
        move(path, dest)
        
    f.close()
    
    
    
    target_folder='IMagenet/tiny-imagenet-200/train/'
    train_folder='IMagenet/tiny-imagenet-200/train_data/'
    test_folder='IMagenet/tiny-imagenet-200/test_data/'
    os.mkdir(train_folder)
    os.mkdir(test_folder)
    paths=glob.glob('IMagenet/tiny-imagenet-200/train/*')
    
    for path in paths:
        folder=path.split('/')[-1].split('\\')[-1]
        source=target_folder+str(folder+'/images/')
        train_final=train_folder+str(folder+'/')
        test_final=test_folder+str(folder+'/')
        os.mkdir(train_final)
        os.mkdir(test_final)
        images=glob.glob(source+str('*'))
    
        random.shuffle(images)
        
        test_images=images[:165].copy()
        train_images=images[165:].copy()
    
        #30%
        for image in test_images:
          file=image.split('/')[-1].split('\\')[-1]
          dest=test_final+str(file)
          move(image,dest)
        
        #70%
        for image in train_images:
          file=image.split('/')[-1].split('\\')[-1]
          dest=train_final+str(file)
          move(image,dest)