import torch
import torch.nn as nn
import numpy as np 
import os
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import models



class MN_NET(torch.nn.Module):
    def __init__(self, batch_size, seq_len=1,num_layers=2,hidden_size = 2, action_categories=0):
        super(MN_NET, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.categories = action_categories
        self.seq_len = seq_len
        self.resnet = models.resnet101(pretrained = True)
        #z = torch.rand([1,3,224,224])
        modules=list(self.resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.linear_layer = torch.nn.Linear(in_features = 2048*7*7, out_features = 2048)
        self.dropout = torch.nn.Dropout(0.4)
        self.linear_layer_two = torch.nn.Linear(in_features = 2048,out_features =256)
        self.linear_classifier = torch.nn.Linear(in_features = self.num_layers*self.hidden_size, out_features= self.categories)
        self.gru = torch.nn.GRU(input_size = 256,hidden_size = 256,num_layers = 2)
        
        
            

        
        
        
    def forward(self, img):
        self.h_init = torch.rand((self.num_layers,self.batch_size,256))
        img = self.resnet(img)
        img = img.view(-1,2048*7*7)

       	img = F.relu(self.linear_layer(img))
       	img = self.dropout(img)
       	img = F.relu(self.linear_layer_two(img))

       	input_seq = torch.autograd.Variable(torch.zeros(self.seq_len,self.batch_size,256), requires_grad = True)
       	for i in range(input_seq.shape[0]):
            input_seq[i,:,:] = img[:,:]

        output, h_out = self.gru(input_seq,self.h_init)

       	h_out = h_out.transpose(0,1)
        print(h_out.size())
       	h_out = h_out.view(-1,self.num_layers*self.hidden_size)
        print(h_out.size())
       	h_out = F.softmax(self.linear_classifier(h_out))

       	return output,h_out


if __name__ == '__main__':
    mn_obj = MN_NET(1,1,action_categories=3)
    z = torch.rand([1,3,224,224])
    out, hout = mn_obj.forward(z)
    print(out.size(), hout.size())






        		






