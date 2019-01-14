import torch
import torch.nn as nn
import numpy as np 
import os
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F 
from torchvision import models, transforms
import sys
import sys.path
from torch.nn.utils.rnn import pack_padded_sequence 
#import nltk
#from nltk.corpus import stopwords
#from nltk import tokenize

MODEL_PATH = ''



class MN_NET(torch.nn.Module):
    def __init__(self, batch_size, seq_len=1,hidden_size = 2,embed_size = 1,vocab_size = 1, action_categories=0):
        super(MN_NET, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.embed_size = embed_size
        self,vocab_size = vocab_size
        #self.num_layers = num_layers
        self.categories = action_categories
        
        self.max_seq_len = seq_len
        self.resnet = models.resnet101(pretrained = True)
        modules=list(self.resnet.children())[:-2]
        self.resnet=nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
       
        self.linear_layer = nn.Linear(in_features = 2048*7*7, out_features = 2048)
        self.dropout = nn.Dropout(0.4)
        self.linear_layer_embed = nn.Linear(in_features = 2048,out_features = self.embed_size)
        self.linear_classifier = nn.Linear(in_features = self.hidden_size, out_features= self.categories)

        self.embed_layer = nn.Embedding(self.vocab_size,self.embed_size)
        self.gru = nn.GRU(input_size = self.embed_size ,hidden_size = self.hidden_size,num_layers = 1,batch_first = True)
        self.bn = nn.BatchNorm1d(self.embed_size, momentum = 0.01)
        self.linear_to_vocab = nn.Linear(in_features=self.hidden_size,out_features= self.vocab_size)
        
        
        
        
        
    def forward(self, img, captions,lengths):
        self.h_init = torch.rand((1,batch_size,self.hidden_size))
        img = self.resnet(img)
        img = img.view(-1,2048*7*7)

       	img = F.relu(self.linear_layer(img))
       	img = self.dropout(img)
       	img = self.linear_layer_embed(img)
        features = self.bn(img)
        embeddings = self.embed_layer(captions)
        embeddings = torch.cat((features.unsqueeze(1),embeddings), 1)

       	packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        outputs, _ = self.gru(packed,self.h_init)

       	"""h_out = h_out.transpose(0,1)
        print(h_out.size())
       	h_out = h_out.view(-1,self.num_layers*self.hidden_size)
        print(h_out.size())
       	h_out = F.softmax(self.linear_classifier(h_out))
        """
        outputs_fin = self.linear_to_vocab(outputs[0])

       	return outputs_fin




def train(model,data_loader,epochs = 10,checkpoint = 10):

  model.train()
  loss_criterion = nn.CrossEntropyLoss()
  
  optimizer = optim.Adam(model.parameters(),lr = 0.0002)
  total_step = len(data_loader)

  for epoch in range(epochs):

    for i,(images,captions,lengths) in enumerate(data_loader):

       targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
       outputs = model(images,captions,lengths)
       loss = loss_criterion(outputs,targets)

       model.zero_grad()
       loss.backward()
       optimizer.step()

       if i%5==0:
        print('Epoch--->[{}/{}], Step--->[{}/{}], Loss--->{}'.format(epoch+1,epochs,i,total_step,loss.item()))

    if (epoch+1)%checkpoint==0:
      torch.save(model.state_dict(),MODEL_PATH)

  

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





        		






