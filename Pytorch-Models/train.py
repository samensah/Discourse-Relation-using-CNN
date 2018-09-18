import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from torch.autograd import Variable

import warnings
warnings.filterwarnings('always')


def _generate_batch(data, batch_size=200, no_shuffles=1):
    # generate a batch on data = dataset[train_data], dataset[test_data],...
    size = len(data['arg1'])
    # shuffle at first
    for i in range(no_shuffles):
        for cur in range(size):
            target = np.random.randint(cur, size)
            if target != cur:
                for k in data:
                    tmp = data[k][target].copy()
                    data[k][target] = data[k][cur]
                    data[k][cur] = tmp
    nb_batch = (size + batch_size - 1) // batch_size
    for index in range(nb_batch):
        begin, end = index * batch_size, min((index + 1) * batch_size, size)
        cur_data = {}
        for k in data:
            # k is arg1, arg2, argplus, sense ...
            cur_data[k] = data[k][begin:end]
        yield (cur_data)


def _prepare_inputs(data_batched):
    '''Inputs for the model'''
    inputs = []
    inputs.append(data_batched['arg1'])
    inputs.append(data_batched['pos1'])
    inputs.append(data_batched['arg2'])
    inputs.append(data_batched['pos2'])
    return [inputs[0], inputs[1], inputs[2], inputs[3]]


def train(train_data, test_data, batch_size, model, args):
    if args.cuda:
        model.cuda()
    print(model.parameters)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in _generate_batch(train_data, batch_size):
            # get inputs for training
            arg1, pos1, arg2, pos2 = _prepare_inputs(batch)
            # transform target into 1D tensor
            target = np.array([np.nonzero(x)[0] for x in batch['sense']])
            target = torch.from_numpy(target).type(torch.LongTensor).squeeze(1)
            # transform data to tensors
            arg1, pos1, arg2, pos2 = torch.from_numpy(arg1).type(torch.LongTensor), torch.from_numpy(pos1).type(torch.LongTensor), \
                                     torch.from_numpy(arg2).type(torch.LongTensor), torch.from_numpy(pos2).type(torch.LongTensor)


            if args.cuda:
                target = target.cuda()
                arg1, pos1, arg2, pos2 = arg1.cuda(), pos1.cuda(), arg2.cuda(), pos2.cuda()

            optimizer.zero_grad()
            if args.pos:
                logit = model(arg1, pos1, arg2, pos2)
            else:
                logit = model(arg1, arg2)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                # calculate accuracy
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch_size
                # calculate fscore
                logits = torch.max(logit, 1)[1].numpy()
                fscore = f1_score(target.numpy(), logits, average='macro')

                # write results
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f} - fscore: {:.3f} - acc: {:.2f}% ({}/{})'.format(steps, loss.item(), fscore,
                                                                             accuracy, corrects, batch_size))

            if steps % args.test_interval == 0:
                fscore = eval(test_data, model, args)
                if fscore > best_acc:
                    best_acc = fscore
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(test_data, model, args):
    model.eval()
    # get inputs for training
    arg1, pos1, arg2, pos2 = _prepare_inputs(test_data)
    # transform target into 1D tensor
    target = np.array([np.nonzero(x)[0] for x in test_data['sense']])
    target = torch.from_numpy(target).type(torch.LongTensor).squeeze(1)
    # transform data to tensors
    arg1, pos1, arg2, pos2 = torch.from_numpy(arg1).type(torch.LongTensor), \
                             torch.from_numpy(pos1).type(torch.LongTensor), \
                             torch.from_numpy(arg2).type(torch.LongTensor), \
                             torch.from_numpy(pos2).type(torch.LongTensor)

    if args.cuda:
        target = target.cuda()
        arg1, pos1, arg2, pos2 = arg1.cuda(), pos1.cuda(), arg2.cuda(), pos2.cuda()

    if args.pos:
        logit = model(arg1, pos1, arg2, pos2)
    else:
        logit = model(arg1, arg2)

    loss  = F.cross_entropy(logit, target, size_average=False)

    # calculate fscore
    logits = torch.max(logit, 1)[1].numpy()
    fscore = f1_score(target.numpy(), logits, average='macro')

    print('\nEvaluation - loss: {:.6f} -  fscore: {:.4f} \n'.format(loss.item(), fscore))
    return fscore


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)