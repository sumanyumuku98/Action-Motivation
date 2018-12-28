#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 00:01:17 2018

@author: sumanyu
"""

import numpy as np
import torch
import os
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AR_NET(nn.Module):
    def __init__(self,K):
        super(AR_NET,self).__init__()
        self.k = K
        self.resnet = models.resnet101(pretrained = True)
        #z = torch.rand([1,3,224,224])
        modules=list(self.resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        #print(self.resnet(z).shape)

        
        self.b = nn.Linear(2048,1, bias = True)
        self.a = nn.Linear(2048,self.k,bias = True)
     
    def forward(self,z):
        
        z = self.resnet(z)
        tensor_list = list()
        for x in z:
            xb = F.relu(self.b((x.view(-1,7*7).transpose(0,1))))
            #print(xb.shape)
            xa = F.relu(self.a((x.view(-1,7*7).transpose(0,1))))
            xa = xa.transpose(0,1)
            #print(xa.shape)

            val = torch.mm(xa,xb)
            val = torch.squeeze(val)
            #print(type(val),val.shape)
            tensor_list.append(val)
            
        z = torch.stack(tensor_list)
        #print(z.shape)
        return F.softmax(z,dim = 1)
    

def train(model,train_loader,epochs = 100):
    model.train()
    optimizer = optim.Adam(model.parameters(),lr = 0.0001)
    loss_criterion = F.cross_entropy()
    train_loss = []
    for epoch in range(epochs):
        batch_loss =[]
        for batch_i, (data,target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = loss_criterion(data,target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss)
            if epoch%50==0:
                print("Epoch:{}--> Loss:{}".format(epoch,loss))
                
        epoch_loss = np.mean(np.array(batch_loss) )
        train_loss.append(epoch_loss)
        
    return train_loss    
    
    
            

def preprocess_img(image):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    return preprocess(image)

if __name__ == '__main__':
    a = AR_NET(3)
    z = torch.rand([1,3,224,224])
    print("hello")
    print(a.forward(z))


