# -*- coding: utf-8 -*-

import time
import torch
import numpy as np
import torch.nn as nn
import argparse
import data_load
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from data_load import *


class Net(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes,
                 network_name,
                 group_list):
        super(Net,self).__init__()
        self.group_list = group_list
        self.network_name = network_name
        self.l1 = nn.Linear(in_feats, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 32)
        self.l5 = nn.Linear(32, 16)
        self.l6 = nn.Linear(16, 8)
        self.classify = nn.Linear(8, n_classes)
        # self.l1 = nn.Linear(in_feats, 64)
        # self.l2 = nn.Linear(64, 32)
        # self.l3 = nn.Linear(32, 16)
        # self.l4 = nn.Linear(16, 8)
        # self.classify = nn.Linear(8, n_classes)

    def forward(self, x, Aggregate_mode):
        print("DNN forward")
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.l6(x))
        if Aggregate_mode == 'average':
            newh = logits2Grouplogits(x, self.group_list, self.network_name)
        elif Aggregate_mode == 'convolution':
            newh = AggregateLogitsWithConvolution(self.network_name, self.group_list, x)
        return self.classify(newh)


def evaluate(model, features, labels, nid, aggregate_mode):
    model.eval()
    with torch.no_grad():         
        logits = model(features, aggregate_mode)
        logits = logits[nid]
        labels = labels[nid]
        print(logits[0:30])
        _, indices = torch.max(logits, dim=1)
        # print(torch.max(logits, dim=1))
        # print(indices)

        correct = torch.sum(indices == labels)
        # print(indices)
        # print(labels)

        TP = torch.sum(indices & labels)

        FP = torch.sum(indices & (1 - labels))
        # print(correct,TP, FP)


        accuracy = correct.item() * 1.0 / len(labels)
        precision = TP * 1.0 / (TP + FP)
        recall = TP / torch.sum(labels)
        F1_score = 2 / (1 / precision + 1 / recall)
        # precision, recall, F1_score, _ = precision_recall_fscore_support(labels, indices, average="binary")

        return accuracy, precision, recall, F1_score


def main(args):
    print("main begin")
    g, GroupAndLabel = data_load(args.network_name, args.label_rate, args.train_rate, args.label_ratio)
    features = g.ndata['feat']
    labels = GroupAndLabel['label']
    train_mask = GroupAndLabel['train_mask']
    test_mask = GroupAndLabel['test_mask']
    group_rank = GroupAndLabel['group_rank']
    in_feats = features.shape[1]
    n_classes = 2
    # n_edges = g.number_of_edges()
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


    # #Initialize the model
    model = Net(in_feats, n_classes, args.network_name, group_rank)
    if cuda:
        model.cuda()
        # use optimizer
    loss_fcn = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
    # # Define loss criterion
    # criterion = F.cross_entropy
    # # Define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, args.aggregate_mode)
        print(logits[:2])
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        # print(train_mask,train_nid)
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
    # print(test_mask[0:30])
    test_nid = test_mask.nonzero().squeeze()
    # print(test_nid[0:30])
    acc, pre, recall, F1_score = evaluate(model, features, labels, test_nid, args.aggregate_mode)
    print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))

#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='GraphSAGE')
#
#     parser.add_argument("network_name", type=str,
#                         help="get network name")
#
#     parser.add_argument("--label_rate", type=float, default=0.05,
#                         help="label rate")
#     parser.add_argument("--label_ratio", type=float, default=2,
#                         help="ratio between negative and positive labels")
#
#     parser.add_argument("--train_rate", type=float, default=0.7,
#                         help="train set rate")
#     parser.add_argument("--test_rate", type=float, default=0.3,
#                         help="test set rate")
#
#     parser.add_argument("--dropout", type=float, default=0.4,
#                         help="dropout probability")
#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2,
#                         help="learning rate")
#     parser.add_argument("--n-epochs", type=int, default=1000,
#                         help="number of training epochs")
#     parser.add_argument("--weight-decay", type=float, default=5e-4,
#                         help="Weight for L2 loss")
#     args = parser.parse_args()
#
#     main(args)