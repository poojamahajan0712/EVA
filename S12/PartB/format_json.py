# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:35:21 2020

@author: pooja
"""

import pandas as pd 
import json
  
def formatted_out(filename='via_export_coco_19apr.json'):
    with open(filename) as json_data:
        data = json.load(json_data)
    
    
    
    img_id,file_name,width,height=([] for i in range(4))
    for i in range(len(data['images'])):
        img_id.append(data['images'][i]['id'])
        file_name.append(data['images'][i]['file_name'])
        width.append(data['images'][i]['width'])
        height.append(data['images'][i]['height'])
    
    df = pd.DataFrame(list(zip(img_id,file_name,width,height)), columns =['img_id','file_name','img_width','img_height']) 
    
    #del img_id,file_name,width,height
    
    
    img_id,a1,a2,a3,a4=([] for i in range(5))
    
    for i in range(len(data['annotations'])):
        img_id.append(data['annotations'][i]['id'])
        a1.append(data['annotations'][i]['bbox'][0])
        a2.append(data['annotations'][i]['bbox'][1])
        a3.append(data['annotations'][i]['bbox'][2])
        a4.append(data['annotations'][i]['bbox'][3])
    
    df1 = pd.DataFrame(list(zip(img_id,a1,a2,a3,a4)), columns =['img_id','x','y','b_width','b_height']) 
    
    #del img_id,a1,a2,a3,a4
    
    comb=pd.merge(df,df1,how='left',on='img_id')
    #del df1,df
    #comb.to_csv('annotations.csv',index=False)
    
    
    comb['new_b_height']=comb['b_height']/comb['img_height']
    comb['new_b_width']=comb['b_width']/comb['img_width']
    
    X=comb[['new_b_height','new_b_width']]
    return(X)