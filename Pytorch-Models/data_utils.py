import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import pickle
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

seed_num = 233

torch.manual_seed(seed_num)
random.seed(seed_num)


class Data(object):
    def __init__(self, datafile, which_data='train_data', shuffle=True):

        # dataset attributes
        self.datafile  = datafile
        self.dataset   = pickle.load(open(self.datafile, 'rb'))

        self.word_WE    = Variable(torch.from_numpy(self.dataset['word_WE']), requires_grad = True)
        self.pos_WE     = Variable(torch.from_numpy(self.dataset['pos_WE']),  requires_grad = True)

        if shuffle == True:
            perm = [i for i in range(len(self.dataset[which_data]['arg1']))]
            np.random.shuffle(perm)
        else:
            perm = [i for i in range(len(self.dataset[which_data]['arg1']))]

        if which_data == 'train_data':
            self.arg1 = torch.from_numpy(self.dataset['train_data']['arg1'][perm]).long()
            self.arg2 = torch.from_numpy(self.dataset['train_data']['arg2'][perm]).long()
            self.pos1 = torch.from_numpy(self.dataset['train_data']['pos1'][perm]).long()
            self.pos2 = torch.from_numpy(self.dataset['train_data']['pos2'][perm]).long()
            self.arg2plus = torch.from_numpy(self.dataset['train_data']['arg2plus'][perm]).long()
            self.pos2plus = torch.from_numpy(self.dataset['train_data']['pos2plus'][perm]).long()
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['train_data']['sense']])
            self.sense    = torch.from_numpy(sense[perm]).long()

        elif which_data == 'dev_data':
            self.arg1 = torch.from_numpy(self.dataset['dev_data']['arg1'][perm]).long()
            self.arg2 = torch.from_numpy(self.dataset['dev_data']['arg2'][perm]).long()
            self.pos1 = torch.from_numpy(self.dataset['dev_data']['pos1'][perm]).long()
            self.pos2 = torch.from_numpy(self.dataset['dev_data']['pos2'][perm]).long()
            self.arg2plus = torch.from_numpy(self.dataset['dev_data']['arg2plus'][perm]).long()
            self.pos2plus = torch.from_numpy(self.dataset['dev_data']['pos2plus'][perm]).long()
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['dev_data']['sense']])
            self.sense    = torch.from_numpy(sense[perm]).long()

        elif which_data == 'test_data':
            self.arg1 = torch.from_numpy(self.dataset['test_data']['arg1'][perm]).long()
            self.arg2 = torch.from_numpy(self.dataset['test_data']['arg2'][perm]).long()
            self.pos1 = torch.from_numpy(self.dataset['test_data']['pos1'][perm]).long()
            self.pos2 = torch.from_numpy(self.dataset['test_data']['pos2'][perm]).long()
            self.arg2plus = torch.from_numpy(self.dataset['test_data']['arg2plus'][perm]).long()
            self.pos2plus = torch.from_numpy(self.dataset['test_data']['pos2plus'][perm]).long()
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['test_data']['sense']])
            self.sense    = torch.from_numpy(sense[perm]).long()

        else:
            print('which_data is either "train_data", "dev_data" or "test_data"')


#train_data = Data(datafile='temporal_data.pic', which_data='train_data', shuffle=False)
#print(train_data.sense)