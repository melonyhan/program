# -*- coding: utf-8 -*-
import argparse
import GraphSAGE
import GAT
import GCN
import DNN
import node2vec_new


def main(args):
    if args.model_name == 'GraphSAGE':
        GraphSAGE.main(args)  
    elif args.model_name == 'GAT':
        GAT.main(args)  
    elif args.model_name == 'GCN':
        GCN.main(args)
    elif args.model_name == 'DNN':
        DNN.main(args)
    elif args.model_name == 'node2vec_new':
        node2vec_new.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train parameters')
    # 训练相关参数
    parser.add_argument("network_name", type=str,
                        help="get network name")
    # 选择group聚合策略
    parser.add_argument("--aggregate_mode", type=str, default='average',    
                        help="choose a group aggregation strategy")
    parser.add_argument("--model_name", type=str, default='DNN',
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
    parser.add_argument("--lr", type=float, default=2e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
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

    # Parses the node2vec arguments.
    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='../emb/Node2vec.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
    args = parser.parse_args()
    
    main(args)




