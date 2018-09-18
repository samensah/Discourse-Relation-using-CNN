#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import pickle
from models import CGNN_Model, CNNmlp_Model, CNNHighway_Model, GRN_MLP_Model
#import model
import train
#import torchtext.data as data
#import torchtext.datasets as datasets


parser = argparse.ArgumentParser(description='Implicit Discourse Classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-pos', action='store_true', default=True, help='use POS')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=350, help='number of embedding dimension [default: 350]') # embed_dim=300+50
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,2,2',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# load data
print("\nLoading data...")
dataset = pickle.load(open("data/temporal_data.pic", 'rb'))
train_data = dataset['train_data']
test_data  = dataset['test_data']
dev_data   = dataset['dev_data']

# update args and print
args.class_num = 2
args.cuda = (not args.no_cuda) and torch.cuda.is_available();
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
args.pos = False
#args.embed_dim = 300
#model = CNNHighway_Model(args, dataset)         # change the model to use here
model = GRN_MLP_Model(dataset)

if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    model = model.cuda()


# train or predict
if args.test:
    try:
        train.eval(test_data, model, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_data, test_data, args.batch_size, model, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')