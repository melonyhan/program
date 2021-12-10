# encoding: utf-8

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import data_load
from sklearn.metrics import precision_recall_fscore_support


# 参考 https://blog.csdn.net/weixin_45613751/article/details/104092220
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))



class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)
        
    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
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
        FP = torch.sum(indices & (1 - labels))
        accuracy = correct.item() * 1.0 / len(labels)
        precision = TP * 1.0 / (TP + FP)
        recall = TP / torch.sum(labels)
        F1_score = 2 / (1 / precision + 1 / recall)
        # precision, recall, F1_score, _ = precision_recall_fscore_support(labels, indices, average="binary")
        
        return accuracy, precision, recall, F1_score



def main(args):
    g = data_load.data_load(args.network_name, args.label_rate, args.train_rate, args.test_rate,args.label_ratio)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = 2
    n_edges = g.number_of_edges()

    print("""----Data statistics------
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    # graph preprocess and calculate normalization factor
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # 选择是否使用预训练模型
    if args.mode == 'train':
        if args.option == True:
            model = torch.load('./pretrain/{}_{}.pt'.format(args.pretrain_network_name, args.model_name))
        else:
            model = GAT(g, in_feats, args.gat_hidden, n_classes, args.nheads)
    else:
        # create GAT model
        model = GAT(g, in_feats, args.gat_hidden, n_classes, args.nheads)

    if cuda:
        model.cuda()

    # use optimizer
    loss_fcn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                 n_edges / np.mean(dur) / 1000))
                                                
    if args.mode == 'pretrain':
        # save the model 
        torch.save(model, './pretrain/{}_{}.pt'.format(args.network_name, args.model_name))
    
    elif args.mode == 'train':
        test_nid = test_mask.nonzero().squeeze()

        acc, pre, recall, F1_score = evaluate(model, features, labels, test_nid)
        print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='GAT')

#     parser.add_argument("network_name", type=str,
#                         help="get network name")
#     # 训练集参数
#     parser.add_argument("--label_rate", type=float, default=0.2,    
#                         help="label rate")
#     parser.add_argument("--label_ratio", type=float, default=2,    
#                         help="ratio between negative and positive labels")

#     parser.add_argument("--train_rate", type=int, default=0.6,
#                         help="train set rate")
#     parser.add_argument("--test_rate", type=int, default=0.2,
#                         help="test set rate")
#     # GAT模型参数
#     parser.add_argument("--alpha", type=float, default=0.2,
#                         help="alpha of gat")
#     parser.add_argument("--nheads", type=int, default=8,
#                         help="number of heads")
#     parser.add_argument("--hidden", type=int, default=256,
#                         help="number of hidden gat units")
#     # parser.add_argument("--dropout", type=float, default=0.5,
#     #                     help="dropout probability")
#     parser.add_argument("--gpu", type=int, default=-1,
#                         help="gpu")
#     parser.add_argument("--lr", type=float, default=1e-2,
#                         help="learning rate")
#     parser.add_argument("--n_epochs", type=int, default=200,
#                         help="number of training epochs")
#     # parser.add_argument("--n_layers", type=int, default=2,
#     #                     help="number of gat layers")
#     parser.add_argument("--weight_decay", type=float, default=5e-4,
#                         help="Weight for L2 loss")
#     args = parser.parse_args()
    
#     main(args)
