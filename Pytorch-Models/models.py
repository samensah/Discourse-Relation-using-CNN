import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import Cgnn, HighwayMLP
from grn_layer import GatedRelevanceNetwork


class CGNN_Model(nn.Module):
    
    def __init__(self, args, dataset):
        super(CGNN_Model, self).__init__()
        self.args = args

        D  = args.embed_dim    #300+50
        C  = args.class_num    #2
        Ci = 1
        Co = args.kernel_num   #1024
        Ks = args.kernel_sizes #[2,2,2]

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['word_WE']))
        self.pos_embed  = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['pos_WE']))

        self.convs1  = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        #self.dropout = nn.Dropout(args.dropout)

        self.cgnn_layer = Cgnn(input_size=2*len(Ks)*Co)

        self.fc1        = nn.Linear(2*len(Ks)*Co, C)



    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, arg1, pos1, arg2, pos2):
        # projection layer
        arg1 = self.word_embed(arg1)  # (N, W, D)
        arg2 = self.word_embed(arg2)
        pos1 = self.pos_embed(pos1)
        pos2 = self.pos_embed(pos2)
        # concatenate arg and pos
        Arg1 = Variable(torch.cat([arg1, pos1], -1))
        Arg2 = Variable(torch.cat([arg2, pos2], -1))
        # convolutional layer
        args_repr = []
        for x in [Arg1, Arg2]:
            x = x.unsqueeze(1)  # (N, Ci, W, D)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            args_repr.append(x)
        # concatenate for 1 representation for classification
        Arg = torch.cat(args_repr, 1)

        # cgnn layer and classification
        Arg = self.cgnn_layer(Arg)
        logit = self.fc1(Arg)
        return logit

class CNNmlp_Model(nn.Module):

    def __init__(self, args, dataset):
        super(CNNmlp_Model, self).__init__()
        self.args = args

        D = args.embed_dim  # 300+50
        C = args.class_num  # 2
        Ci = 1
        Co = args.kernel_num  # 1024
        Ks = args.kernel_sizes  # [2,2,2]

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['word_WE']))
        self.pos_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['pos_WE']))

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, arg1, pos1, arg2, pos2):
        # projection layer
        arg1 = self.word_embed(arg1)  # (N, W, D)
        arg2 = self.word_embed(arg2)
        pos1 = self.pos_embed(pos1)
        pos2 = self.pos_embed(pos2)
        # concatenate arg and pos
        Arg1 = Variable(torch.cat([arg1, pos1], -1))
        Arg2 = Variable(torch.cat([arg2, pos2], -1))
        # convolutional layer
        args_repr = []
        for x in [Arg1, Arg2]:
            x = x.unsqueeze(1)  # (N, Ci, W, D)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            args_repr.append(x)
        # concatenate for 1 representation for classification
        Arg = torch.cat(args_repr, 1)
        # dropout and classification
        Arg = self.dropout(Arg)  # (N, len(Ks)*Co)
        logit = self.fc1(Arg)  # (N, C)
        return logit


class CNNHighway_Model(nn.Module):

    def __init__(self, args, dataset):
        super(CNNHighway_Model, self).__init__()
        self.args = args

        D = args.embed_dim  # 300+50
        C = args.class_num  # 2
        Ci = 1
        Co = args.kernel_num  # 1024
        Ks = args.kernel_sizes  # [2,2,2]

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['word_WE']))
        self.pos_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['pos_WE']))

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        # self.dropout = nn.Dropout(args.dropout)

        self.highway_layer = HighwayMLP(input_size=2 * len(Ks) * Co)

        self.fc1 = nn.Linear(2 * len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, arg1, pos1, arg2, pos2):
        # projection layer
        arg1 = self.word_embed(arg1)  # (N, W, D)
        arg2 = self.word_embed(arg2)
        pos1 = self.pos_embed(pos1)
        pos2 = self.pos_embed(pos2)
        # concatenate arg and pos
        Arg1 = Variable(torch.cat([arg1, pos1], -1))
        Arg2 = Variable(torch.cat([arg2, pos2], -1))
        # convolutional layer
        args_repr = []
        for x in [Arg1, Arg2]:
            x = x.unsqueeze(1)  # (N, Ci, W, D)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
            x = torch.cat(x, 1)
            args_repr.append(x)
        # concatenate for 1 representation for classification
        Arg = torch.cat(args_repr, 1)

        # cgnn layer and classification
        Arg   = self.highway_layer(Arg)
        logit = self.fc1(Arg)
        return logit


class GRN_MLP_Model(nn.Module):

    def __init__(self, dataset):
        super(GRN_MLP_Model, self).__init__()

        D = 300   #embed for only args
        C = 2     #num of classes 2
        k = 2     #slices in grn in bilinear model
        arg_idx_size = 80  # num of words per arg
        pool_size = 3
        self.hidden_dim = D

        # calculate feature size
        out_dim   = int(arg_idx_size/pool_size)
        fc1_in_feat  = out_dim*out_dim
        fc1_out_feat = int(fc1_in_feat/2)
        fc2_in_feat  = fc1_out_feat
        output       = C

        self.word_embed = nn.Embedding.from_pretrained(torch.FloatTensor(dataset['word_WE']))
        self.bilstm     = nn.LSTM(D, self.hidden_dim // 2, num_layers=1,  bidirectional=True, bias=False)
        self.model_grn  = GatedRelevanceNetwork(output_dim=k, embed_dim=D)
        self.maxpool    = nn.MaxPool2d(pool_size, pool_size)
        self.fc1        = nn.Linear(fc1_in_feat, fc1_out_feat)   # calculate the dimension of input
        self.fc2        = nn.Linear(fc2_in_feat, output)
        self.activate   = nn.functional.softmax


    def forward(self, arg1, arg2):
        batch_size = arg1.size()[0]
        # projection layer
        arg1_embed = self.word_embed(arg1)  # (N, W, D)
        arg2_embed = self.word_embed(arg2)

        # bilstm of input
        arg1_lstm_out, _ = self.bilstm(arg1_embed)
        arg2_lstm_out, _ = self.bilstm(arg2_embed)

        score_matrix = self.model_grn(arg1_lstm_out, arg2_lstm_out)
        pool_scores  = self.maxpool(score_matrix)
        pool_scores  = pool_scores.view(batch_size, -1) # flatten scores

        fc1_out = self.fc1(pool_scores)
        logits  = self.fc2(fc1_out)
        logits  = self.activate(logits, dim=1)
        return logits


#import pickle
#dataset = pickle.load(open("data/temporal_data.pic", 'rb'))
#train_data = dataset['train_data']
#arg1 = torch.Tensor(train_data['arg1'][0:6]).type(torch.LongTensor)
#arg2 = torch.Tensor(train_data['arg2'][0:6]).type(torch.LongTensor)
#model = GRN_MLP_Model(dataset)
#arg1 = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).type(torch.LongTensor)
#arg2 = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]).type(torch.LongTensor)
#print(model)
#model(arg1, arg2)