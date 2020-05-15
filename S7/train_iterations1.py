# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:15:36 2020

@author: pooja
"""

from tqdm import tqdm

def train(n_epoch,trainloader,device,optimizer,net,criterion):
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print("EPOCH:", epoch)
        pbar = tqdm(trainloader)
        correct = 0
        #processed = 0
        #train_loss = 0.0
        for i, data in enumerate(pbar):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            #processed += len(data)

            #pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i} Accuracy={correct/processed:0.2f}')
            #train_acc.append(100*correct/processed)
        #train_loss /= len(trainloader.dataset)
        print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(trainloader.dataset),
        100. * correct / len(trainloader.dataset)))
    print('Finished Training')