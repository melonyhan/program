import argparse
import time
import numpy as np
import data_load
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(g, model, features, labels, nid):
    model.eval()
    with torch.no_grad():         
        logits = model(g, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)  

        correct = torch.sum(indices == labels)
        TP = torch.sum(indices & labels)
        FP = torch.sum(indices & (1 - labels))
        accuracy = correct.item() * 1.0 / len(labels)
        precision = TP * 1.0 / (TP + FP)
        recall = TP / torch.sum(labels)
        F1_score = 2 / (1 / precision + 1 / recall)
        
        return accuracy, precision, recall, F1_score



def main(args):
    g = data_load.data_load(args.network_name, args.label_rate, args.train_rate, args.test_rate, args.label_ratio)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
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

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # 选择是否使用预训练模型
    if args.mode == 'train':
        if args.option == True:
            model = torch.load('./pretrain/{}_{}.pt'.format(args.pretrain_network_name, args.model_name))
        else:
            model = GraphSAGE(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout,
                        args.aggregator_type)
    else:
        # create GraphSAGE model
        model = GraphSAGE(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout,
                        args.aggregator_type)

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}".format(epoch, np.mean(dur), loss.item()))
    if args.mode == 'pretrain':
        # save the model 
        torch.save(model, './pretrain/{}_{}.pt'.format(args.network_name, args.model_name))
    
    elif args.mode == 'train':
        model = model.cpu()
        features = features.cpu()
        labels = labels.cpu()
        test_nid = test_nid.cpu()
        g = g.to(torch.device('cpu'))

        acc, pre, recall, F1_score = evaluate(g, model, features, labels, test_nid)
        print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))





# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='GraphSAGE')

#     parser.add_argument("network_name", type=str,
#                         help="get network name")

#     parser.add_argument("--label_rate", type=float, default=0.05,    
#                         help="label rate")
#     parser.add_argument("--label_ratio", type=float, default=2,    
#                         help="ratio between negative and positive labels")

#     parser.add_argument("--train_rate", type=float, default=0.7,
#                         help="train set rate")
#     parser.add_argument("--test_rate", type=float, default=0.3,
#                         help="test set rate")

#     parser.add_argument("--dropout", type=float, default=0.4,
#                         help="dropout probability")
#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2,
#                         help="learning rate")
#     parser.add_argument("--n-epochs", type=int, default=200,
#                         help="number of training epochs")
#     parser.add_argument("--n-hidden", type=int, default=512,
#                         help="number of hidden GraphSAGE units")
#     parser.add_argument("--n-layers", type=int, default=2,
#                         help="number of hidden GraphSAGE layers")
#     parser.add_argument("--weight-decay", type=float, default=5e-4,
#                         help="Weight for L2 loss")
#     parser.add_argument("--aggregator-type", type=str, default="gcn",
#                         help="Aggregator type: mean/gcn/pool/lstm")
#     args = parser.parse_args()
    
#     main(args)
