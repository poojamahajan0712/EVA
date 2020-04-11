# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:18:31 2020

@author: pooja
"""

import matplotlib.pyplot as plt
import numpy as np

def cycle_graph(step_size,cycle):
    k=2*step_size+1
    #generating values for x axis
    x=np.arange(1,k+1)
    for i in range(2,cycle+1):
        p=np.arange(x[-1],x[-1]+k)
        x=np.concatenate([x,p])
    
    #generating values for y axis 
    t1=np.concatenate([np.arange(1,step_size+1),np.arange(step_size+1,0,-1)])
    y=np.tile(t1,cycle)
    
    #line plot
    plt.plot(x,y,linewidth=1.0)

#function call
cycle_graph(7,5)
