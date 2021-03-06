import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import pickle
import torch.nn.init as init

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

        type_data = ["train_data", "dev_data", "test_data"]
        if shuffle is True:
            if which_data in type_data:
                perm = [i for i in range(len(self.dataset[which_data]['arg1']))]
                np.random.shuffle(perm)
            else:
                raise KeyError('which_data is either "train_data", "dev_data" or "test_data"')
        else:
            if which_data in type_data:
                perm = [i for i in range(len(self.dataset[which_data]['arg1']))]
            else:
                raise KeyError('which_data is either "train_data", "dev_data" or "test_data"')

        if which_data == 'train_data':
            self.arg1 = Variable(torch.from_numpy(self.dataset['train_data']['arg1'][perm]).long(), requires_grad = False)
            self.arg2 = Variable(torch.from_numpy(self.dataset['train_data']['arg2'][perm]).long(), requires_grad = False)
            self.pos1 = Variable(torch.from_numpy(self.dataset['train_data']['pos1'][perm]).long(), requires_grad = False)
            self.pos2 = Variable(torch.from_numpy(self.dataset['train_data']['pos2'][perm]).long(), requires_grad = False)
            self.arg2plus = Variable(torch.from_numpy(self.dataset['train_data']['arg2plus'][perm]).long(), requires_grad = False)
            self.pos2plus = Variable(torch.from_numpy(self.dataset['train_data']['pos2plus'][perm]).long(), requires_grad = False)
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['train_data']['sense']])
            self.sense    = Variable(torch.from_numpy(sense[perm]).long())

        elif which_data == 'dev_data':
            self.arg1 = Variable(torch.from_numpy(self.dataset['dev_data']['arg1'][perm]).long(), requires_grad = False)
            self.arg2 = Variable(torch.from_numpy(self.dataset['dev_data']['arg2'][perm]).long(), requires_grad = False)
            self.pos1 = Variable(torch.from_numpy(self.dataset['dev_data']['pos1'][perm]).long(), requires_grad = False)
            self.pos2 = Variable(torch.from_numpy(self.dataset['dev_data']['pos2'][perm]).long(), requires_grad = False)
            self.arg2plus = Variable(torch.from_numpy(self.dataset['dev_data']['arg2plus'][perm]).long(), requires_grad = False)
            self.pos2plus = Variable(torch.from_numpy(self.dataset['dev_data']['pos2plus'][perm]).long(), requires_grad = False)
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['dev_data']['sense']])
            self.sense    = Variable(torch.from_numpy(sense[perm]).long())

        elif which_data == 'test_data':
            self.arg1 = Variable(torch.from_numpy(self.dataset['test_data']['arg1'][perm]).long(), requires_grad = False)
            self.arg2 = Variable(torch.from_numpy(self.dataset['test_data']['arg2'][perm]).long(), requires_grad = False)
            self.pos1 = Variable(torch.from_numpy(self.dataset['test_data']['pos1'][perm]).long(), requires_grad = False)
            self.pos2 = Variable(torch.from_numpy(self.dataset['test_data']['pos2'][perm]).long(), requires_grad = False)
            self.arg2plus = Variable(torch.from_numpy(self.dataset['test_data']['arg2plus'][perm]).long(), requires_grad = False)
            self.pos2plus = Variable(torch.from_numpy(self.dataset['test_data']['pos2plus'][perm]).long(), requires_grad = False)
            sense = np.array([np.where(r == 1)[0][0] for r in self.dataset['test_data']['sense']])
            self.sense    = Variable(torch.from_numpy(sense[perm]).long())

        else:
            raise KeyError('which_data is either "train_data", "dev_data" or "test_data"')


#train_data = Data(datafile='temporal_data.pic', which_data='train_data', shuffle=True)
#print(train_data.sense)
