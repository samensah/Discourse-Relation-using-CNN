import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
else:
    torch.manual_seed(0)

class GRN(nn.Module):
    """"Gated Relevance Network"""
    def __init__(self, output_dim, input_dim):
        # output_dim is the number of k slices
        self.output_dim = output_dim
        self.emb_dim    = input_dim
        self.activation = nn.functional.tanh  # The f function in the formula
        self.gate_activation = nn.functional.sigmoid

        # initialize variables to be trained
        self.Wb = Variable(torch.rand(self.output_dim, self.emb_dim, self.emb_dim))  #bilinear tensor product weight
        self.Wd = Variable(torch.rand(2 * self.emb_dim, self.output_dim))            #Single Layer Network weights
        self.Wg = Variable(torch.rand(2 * self.emb_dim, self.output_dim))            #gate weights
        self.bg = Variable(torch.rand(self.output_dim))                              #bias gate weights
        self.b  = Variable(torch.rand(self.output_dim))                              #general bias
        self.u  = Variable(torch.rand(self.output_dim, 1))                           #channel weights
        super(GRN, self).__init__()


    def forward(self, arg1, arg2):
        # Get the batch size
        batch_size = arg1.size()[0]
        # Usually len1 = len2 = max_seq_length
        # emb_dim = self.emb_dim
        _, len1, emb_dim = arg1.size()
        _, len2, _       = arg2.size()

        # Repeating the matrices to generate all the combinations
        ne1 = arg1.unsqueeze(2).repeat((1, 1, len2, 1))
        ne1 = ne1.view(batch_size, len1 * len2, emb_dim)
        ne2 = arg2.unsqueeze(1).repeat((1, len1, 1, 1))
        ne2 = ne2.view(batch_size, len1 * len2, emb_dim)

        # Repeating the second matrix to use in Bilinear Tensor Product
        ne2_k = ne2.unsqueeze(-1).repeat((1,1,1,self.output_dim))

        # bilinear tensor product btp
        btp = torch.sum(ne2_k * torch.einsum('bxy,iyk->bxik', (ne1, self.Wb)).permute(0, 1, 3, 2), dim=2)
        btp = btp.view(batch_size, len1, len2, self.output_dim)

        # Concatenating inputs to apply Single Layer Network
        e = torch.cat([ne1, ne2], -1)

        # Single Layer Network
        sln = self.activation(torch.einsum('bxy,yk->bxk', (e, self.Wd)))
        sln = sln.view(batch_size, len1, len2, self.output_dim)

        # Gate
        g = self.gate_activation(torch.einsum('bxy,yk->bxk', (e, self.Wg)) + self.bg)
        g = g.view(batch_size, len1, len2, self.output_dim)

        # Gated Relevance Network
        # s = torch.einsum('bixy,yk->bixk', (g*btp + (1-g)*sln + b, u)).view(batch_size, len1, len2)
        s = torch.einsum('bixy,yk->bixk', (g * btp + (1 - g) * sln + self.b, self.u))

        return s

#import pickle
#import numpy as np
#dataset = pickle.load(open("data/temporal_data.pic", 'rb'))
#model = GRN(output_dim=10, input_dim=5)
#arg1 = torch.Tensor(np.array([[[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]], [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]]))
#arg2 = torch.Tensor(np.array([[[3,2,0.5,14,15],[21,13,0,0,1],[0,1,0,0,2]], [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]]))
#print(model)
#print(model(arg1, arg2))