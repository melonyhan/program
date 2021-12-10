# -*- coding: utf-8 -*-

import time
import torch
import numpy as np
import torch.nn as nn
import argparse
import data_load
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

class Net(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 dropout):
        super(Net,self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        self.layers.append(nn.LeakyReLU(inplace=True))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.LeakyReLU(inplace=True))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes)) 
        # self.layers.append(nn.LeakyReLU(inplace=True))
        self.layers.append(nn.Sigmoid())  
        self.dropout = nn.Dropout(dropout) 


    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(h)
        return h


def evaluate(model, features, labels, nid):
    model.eval()
    with torch.no_grad():         
        logits = model(features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)  

        correct = torch.sum(indices == labels)
        TP = torch.sum(indices & labels)
        accuracy = correct.item() * 1.0 / len(labels)
            
        precision, recall, F1_score, _ = precision_recall_fscore_support(labels, indices, average="binary")

        return accuracy, precision, recall, F1_score

def main(args):
    g = data_load.data_load(args.network_name, args.label_rate, args.train_rate, args.test_rate, args.label_ratio)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = 2
    print("""----Data statistics------'
      #Classes %d
      #Train samples %d
      #Test samples %d""" %
          (n_classes,
           train_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    #Initialize the model        
    model = Net(in_feats, args.n_hidden, n_classes, args.n_layers, args.dropout)
    #Define loss criterion
    criterion = F.cross_entropy
    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if cuda:
        model.cuda()

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = criterion(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, np.mean(dur), loss.item()))

    # model = model.cpu()
    # features = features.cpu()
    # labels = labels.cpu()
    # test_nid = test_nid.cpu()

    acc, pre, recall, F1_score = evaluate(model, features, labels, test_nid)
    print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')

    parser.add_argument("network_name", type=str,
                        help="get network name")

    parser.add_argument("--label_rate", type=float, default=0.05,    
                        help="label rate")
    parser.add_argument("--label_ratio", type=float, default=2,    
                        help="ratio between negative and positive labels")

    parser.add_argument("--train_rate", type=float, default=0.7,
                        help="train set rate")
    parser.add_argument("--test_rate", type=float, default=0.3,
                        help="test set rate")

    parser.add_argument("--dropout", type=float, default=0.4,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=400,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    args = parser.parse_args()
    
    main(args)