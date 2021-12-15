# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from data_load import *


class GCN(nn.Module):
    def __init__(self, 
                 g, 
                 network_name, 
                 in_feats, 
                 n_hidden, 
                 n_classes, 
                 n_layers, 
                 activation, 
                 dropout, 
                 group_list):
        print("GCN init")
        super(GCN, self).__init__()
        self.g = g
        self.network_name = network_name
        self.group_list = group_list
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        for i in range(n_layers - 1):
           self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        #self.classify = GraphConv(n_hidden, n_classes)
        self.classify = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

        # self.conv1 = GraphConv(in_feats, n_hidden)
        # #self.dropout1 = nn.Dropout(p=dropout)
        # self.conv2 = GraphConv(n_hidden, n_hidden)
        # #self.dropout2 = nn.Dropout(p=dropout)
        # self.classify = nn.Linear(n_hidden, n_classes)
        #self.classify = GraphConv(n_hidden, n_classes)
    
    def forward(self, features, Aggregate_mode):
        print("GCN forward")
        h = features  # [1005,8]
        for i, layers in enumerate(self.layers):
           if i!=0:
               h = self.dropout(h)
           h = layers(self.g, h)
        # print(h.shape)  # [1005,128]
        if Aggregate_mode == 'average':
            newh = logits2Grouplogits(h,self.group_list,self.network_name)
        elif Aggregate_mode == 'convolution':
            newh = AggregateLogitsWithConvolution(self.network_name, self.group_list, h)
        
        return self.classify(newh)  # [15018,2]
        
        #卷积层以及**函数
        # h = F.relu(self.conv1(self.g,h))
        # h = F.relu(self.conv2(self.g,h))
        # newh = logits2Grouplogits(h,self.group_rank,self.network_name)
        # return self.classify(newh)
        

def evaluate(model, features, labels, nid, aggregate_mode):
    model.eval()
    with torch.no_grad():         
        logits = model(features, aggregate_mode)
        #group_logits = logits2Grouplogits(logits,group_rank,network_name)
        group_logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(group_logits, dim=1)  

        correct = torch.sum(indices == labels)
        TP = torch.sum(indices & labels)
        FP = torch.sum(indices & (1 - labels))
        # print(indices, labels)
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
    labels = GroupAndLabel['label']  # [15018]
    print(labels[0:30])
    train_mask = GroupAndLabel['train_mask']
    test_mask = GroupAndLabel['test_mask']
    group_rank = GroupAndLabel['group_rank']  # [[86, 107, 121, 160, 979], 468]*15018
    in_feats = features.shape[1]  # 8
    n_classes = 2
    n_edges = g.number_of_edges()
    
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           test_mask.int().sum().item()))
    # initialize graph
    
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
    
    # train_nid = train_mask.nonzero().squeeze()
    # val_nid = val_mask.nonzero().squeeze()
    # test_nid = test_mask.nonzero().squeeze()
    # graph preprocess and calculate normalization factor
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    model = GCN(g, args.network_name, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, group_rank)
    if cuda:
        model.cuda()
    # use optimizer
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)    

    model.train()    
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        # forward
        group_logits = model(features, args.aggregate_mode)
        print(group_logits[:2])
        loss = loss_fcn(group_logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                 n_edges / np.mean(dur) / 1000))
    # test the model
    test_nid = test_mask.nonzero().squeeze()
    acc, pre, recall, F1_score = evaluate(model, features, labels, test_nid, args.aggregate_mode)
    print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))

