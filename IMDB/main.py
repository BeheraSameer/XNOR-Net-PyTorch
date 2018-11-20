from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import models
from torchtext import datasets, data
from torch.autograd import Variable
import time
import random

def save_state(model, acc):
    '''
    Saves the best model for future use
    '''

    '''
    print('==> Prints all keys and states. Need to enable binarization to view the binarized data...')
    bin_op.binarization()
    print(model.state_dict())
    bin_op.restore()
    '''
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            } 
    
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/'+args.arch+'.best.pth.tar')


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc



def train(model, iterator, optimizer, criterion):
    print('Training Started') 
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch IMDB Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
            help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
            metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
            help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='CNN',
            help='the IMDB network structure: CNN')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)
    
    torch.manual_seed(args.seed)
    # Is it necessary???
    torch.backends.cudnn.deterministic = True


    if args.cuda:
        device = torch.device("cuda")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")

    text = data.Field(tokenize='spacy')
    label = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text, label)

    train_data, valid_data = train_data.split(random_state=random.seed(args.seed))

    # Build Vocabulary
    text.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
    label.build_vocab(train_data)
    
 
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=args.batch_size, 
        device=device)

    # generate the model
    # TODO: move these parameter initializations into the constructor
    input_dim = len(text.vocab)
    embedding_dim = 100
    n_filters = 100
    filter_sizes = [3,4,5]
    output_dim = 1
    dropout = 0.5
    if args.arch == 'CNN':
        model = models.CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
    else:
        print('ERROR: specified arch is not suppported')
        exit()


    # Load the pretrained embeddings
    pretrained_embeddings = text.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['state_dict'])

    model = model.to(device)
    
    print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    base_lr = 0.1
    
    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'weight_decay': args.weight_decay,
            'key':key}]
    
    optimizer = optim.Adam(params, lr=args.lr,
            weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()
    
    # define the binarization operator
    # bin_op = util.BinOp(model)

    if args.evaluate:
        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        print('| Test Loss: {:.3f} | Test Acc: {:.2f}% |'.format(test_loss, test_acc * 100))
        exit()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
        print('| Epoch: {:02} | Train Loss: {:.3f} | Train Acc: {:.2f}% | Val. Loss: {:.3f} | Val. Acc: {:.2f}% |'.format(epoch+1, train_loss, train_acc*100, valid_loss, valid_acc*100))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print('| Test Loss: {:.3f} | Test Acc: {:.2f}% |'.format(test_loss, test_acc * 100))
