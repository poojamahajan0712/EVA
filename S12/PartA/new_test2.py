# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:44:21 2020

@author: pooja
"""

import torch
import numpy as np

def test(is_last,model, device, test_loader, criterion,test_losses, test_accs,pred_wrong_t,true_wrong_t,image_t):
    model.eval()
    test_loss = 0
    correct = 0
    pred_wrong = []
    true_wrong = []
    image = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss +=criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            is_correct = pred.eq(target.view_as(pred))
            correct += is_correct.sum().item()
            
            
            #preds = (pred.cuda()).cpu().numpy()
            #target = (target.cuda()).cpu().numpy()
        
            
            if(is_last):
              misclassified_inds = (is_correct==0).nonzero()[:,0]
              for i in misclassified_inds:
                #pred1.append(preds[i])
                #true.append(target[i])

                #if(preds[i]!=target[i]):
                    pred_wrong.append(pred[i][0].cpu().numpy())
                    true_wrong.append(target[i].cpu().numpy())
                    image.append(data[i])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    pred_wrong_t.extend(pred_wrong)
    true_wrong_t.extend(true_wrong)
    image_t.extend(image)

