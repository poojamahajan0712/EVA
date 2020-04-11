# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:11:57 2020

@author: pooja
"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Prep Block
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
            
        ) #i/p=32, output_size = 30  Rf 3 Jout - 1

        # Layer 1
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            ) # i/p=30,output_size = 32 RF 5 Jout -1
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
           # nn.ReLU()
            )
        # Layer 2
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        
        # Layer 3
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            ) # i/p=30,output_size = 32 RF 5 Jout -1
        self.r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            #nn.ReLU()
            )
        
        self.pool1 = nn.MaxPool2d(4, 2)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) 


    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        
        #iden=x
        out=self.r1(x)
        out=out+x
        x=F.relu(out)
        
        #x=x+self.r1(x)
        x = self.c3(x)
        x = self.c4(x)
        
        out=self.r2(x)
        out=out+x
        x=F.relu(out)
        
        #x=x+self.r2(x)
        x = self.pool1(x)
        x = self.fc(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)
