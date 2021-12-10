# -*- coding: utf-8 -*-
import argparse
import GraphSAGE
import GAT
import GCN
import DNN


def main(args):
    if args.model_name == 'GraphSAGE':
        GraphSAGE.main(args)  
    elif args.model_name == 'GAT':
        GAT.main(args)  
    elif args.model_name == 'GCN':
        GCN.main(args)
    elif args.model_name == 'DNN':
        DNN.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train parameters')
    # 训练相关参数
    parser.add_argument("network_name", type=str,
                        help="get network name")
    # 选择group聚合策略
    parser.add_argument("--aggregate_mode", type=str, default='average',    
                        help="choose a group aggregation strategy")
    parser.add_argument("--model_name", type=str, default='GCN',
                        help="choose a model")
    parser.add_argument("--label_rate", type=float, default=0.05,    
                        help="label rate")
    parser.add_argument("--label_ratio", type=float, default=3,    
                        help="ratio between negative and positive labels")
    # 训练集和测试集按7 : 3划分
    parser.add_argument("--train_rate", type=float, default=0.7,
                        help="train set rate")
    parser.add_argument("--test_rate", type=float, default=0.3,
                        help="test set rate")

    # GraphSAGE模型参数
    parser.add_argument("--dropout", type=float, default=0.4,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden GraphSAGE units")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden GraphSAGE layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    
    # GAT相关参数
    parser.add_argument("--gat-hidden", type=int, default=16,
                        help="number of hidden gat units")
    parser.add_argument("--nheads", type=int, default=4,
                        help="number of heads")
    parser.add_argument("--fc_hidden_1", type=int, default=16,
                        help="number of hidden fc1 units")
    parser.add_argument("--fc_hidden_2", type=int, default=8,
                        help="number of hidden fc2 units")

    args = parser.parse_args()
    
    main(args)




