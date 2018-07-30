# @Author : bamtercelboo
# @Datetime : 2018/07/19 22:35
# @File :

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import pickle
import torch.nn.init as init
from data_utils import Data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed_num = 233
pad = "<pad>"
unk = "<unk>"

torch.manual_seed(seed_num)
random.seed(seed_num)

"""
    Neural Network: CNN
"""


class CNN_Text(nn.Module):

    def __init__(self, data='temporal'):
        super(CNN_Text, self).__init__()

        self.train_data = Data(datafile=data + '_data.pic', which_data='train_data', shuffle=True)
        self.dev_data   = Data(datafile=data + '_data.pic', which_data='train_data', shuffle=False)
        self.test_data  = Data(datafile=data + '_data.pic', which_data='train_data', shuffle=False)

        C = 2   #num of classes
        Ci = 1
        Co = 1024
        Ks = [2,3,5]

        self.word_ndims = 300
        self.pos_ndims  = 50

        self.word_WE  = torch.FloatTensor(self.train_data.word_WE)
        self.pos_WE   = torch.FloatTensor(self.train_data.pos_WE)

        # Embedding functions
        self.WE_embed = nn.Embedding.from_pretrained(self.word_WE)
        self.WE_embed.weight.requires_grad = True
        self.PE_embed = nn.Embedding.from_pretrained( self.pos_WE)
        self.PE_embed.weight.requires_grad = True

        # "using narrow convolution"
        self.shared_cnn = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, self.word_ndims+self.pos_ndims), bias=True) for K in Ks]

        # for cnn cuda
        args_cud = False
        if args_cud is True:
            for conv in self.shared_cnn:
                conv = conv.cuda()

        # dropout
        self.dropout = nn.Dropout(0.2)
        self.dropout_embed = nn.Dropout(0.2)

        in_features = 2 * len(Ks) * Co  #features from convs for both Arg1, Arg2
        self.fc = nn.Linear(in_features=in_features, out_features=C)





    def forward(self, arg1, arg2, pos1, pos2):
        arg1 = self.WE_embed(arg1)
        arg2 = self.WE_embed(arg2)
        pos1 = self.PE_embed(pos1)
        pos2 = self.PE_embed(pos2)

        Arg1 = torch.cat((arg1, pos1), -1)
        Arg2 = torch.cat((arg2, pos2), -1)

        Arg1 = self.dropout_embed(Arg1)
        Arg2 = self.dropout_embed(Arg2)

        Arg1 = Arg1.unsqueeze(1)  # (N,Ci,W,D)
        Arg2 = Arg2.unsqueeze(1)  # (N,Ci,W,D)

        Arg1 = [F.relu(conv(Arg1)).squeeze(3) for conv in self.shared_cnn]
        Arg1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in Arg1]
        Arg2 = [F.relu(conv(Arg2)).squeeze(3) for conv in self.shared_cnn]
        Arg2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in Arg2]

        Arg1 = torch.cat(Arg1, 1)
        Arg2 = torch.cat(Arg2, 1)

        repre = torch.cat([Arg1, Arg2], 1)

        logit = self.fc(repre)

        return logit

# Train the model
def train(model):
    train_data = model.train_data
    dev_data   = model.dev_data
    test_data  = model.test_data

    # Build loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)

    epochs = 2
    model.train()
    for epoch in range(1, epochs):
        print("\n# The {} Epoch, All {} Epochs ! #".format(epoch, epochs))

        # train data
        logit = model(train_data.arg1, train_data.arg2, train_data.pos1, train_data.pos2)
        loss  = criterion(logit, train_data.sense)
        print('loss: ', loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, epochs, loss.item()))

# Test the model
def test(model):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    test_data  = model.test_data
    with torch.no_grad():
        correct = 0
        total = 0
        test_data.arg1.to(device); test_data.arg2.to(device); test_data.pos1.to(device); test_data.pos2.to(device)
        test_data.sense.to(device)

        logit = model(test_data.arg1, test_data.arg2, test_data.pos1, test_data.pos2)

        _, predicted = torch.max(logit.data, 1)
        total += test_data.sense.size(0)
        correct += (predicted == test_data.sense).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')


model = CNN_Text(data='temporal')
#train(model)
test(model)
