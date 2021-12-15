import time

import torch
from gensim.models import Word2Vec
from data_load import *
from torch import nn


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    # 得到节点序列，每个序列长度为walk_length，共有num_walks*num_noded个序列
    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            # 得到unnormalized_probs列表，其中每个元素是边的权重
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            # normalized_probs得到归一化后的，到各个结点的概率
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# def read_graph():
#     '''
#     Reads the input network in networkx.
#     '''
#     if args.weighted:
#         G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
#     else:
#         G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
#         for edge in G.edges():
#             G[edge[0]][edge[1]]['weight'] = 1
#
#     if not args.directed:
#         G = G.to_undirected()
#
#     return G


def learn_embeddings(walks, args):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = list(list(map(str, walk)) for walk in walks)
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)
    # model.wv.save_word2vec_format(args.output)

    return model


def get_embeddings(w2v_model, graph):
    invalid_word = []
    embeddings = []
    for word in graph.nodes():
        if word in w2v_model.wv:
            embeddings.append(w2v_model.wv[word])
        else:
            invalid_word.append(word)
    print("无效word", len(invalid_word))
    print("有效embedding", len(embeddings))

    return torch.Tensor(embeddings)


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 network_name,
                 group_rank
                 ):
        super(Linear, self).__init__()
        self.network_name = network_name
        self.group_rank = group_rank

        self.classify = nn.Linear(in_features, out_features)

    def forward(self, features, aggregate_mode):
        if aggregate_mode == 'average':
            features = logits2Grouplogits(features, self.group_rank, self.network_name)
        elif aggregate_mode == 'convolution':
            features = AggregateLogitsWithConvolution(self.network_name, self.group_rank, features)
        return self.classify(features)


def evaluate(model, features, labels, nid, aggregate_mode):
    model.eval()
    with torch.no_grad():
        logits = model(features, aggregate_mode)
        # group_logits = logits2Grouplogits(logits,group_rank,network_name)
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
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    # nx_G = read_graph()
    # myargs = myArgs()
    # g, GroupAndLabel = data_load(args.network_name, args.label_rate, args.train_rate, args.label_ratio)
    with open('../Data/networkx/{}-networkx.json'.format(args.network_name), 'r') as fr:
        G_data = json.load(fr)
    G = json_graph.node_link_graph(G_data)
    g = nx.Graph(G)
    print(g)
    for edge in g.edges():
        g[edge[0]][edge[1]]['weight'] = 1
    G = Graph(g, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    # walk_length:对每一个node采样时的步长(80)
    walks = G.simulate_walks(args.num_walks, args.walk_length)  # (10050, 80)
    w2v_model = learn_embeddings(walks, args)
    embeddings = get_embeddings(w2v_model, g)

    _, GroupAndLabel = data_load(args.network_name, args.label_rate, args.train_rate, args.label_ratio)
    group_rank = GroupAndLabel['group_rank']  # [[86, 107, 121, 160, 979], 468]*15018
    labels = GroupAndLabel['label']  # [15018]
    train_mask = GroupAndLabel['train_mask']
    test_mask = GroupAndLabel['test_mask']

    classify = Linear(128, 2, args.network_name, group_rank)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classify.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    classify.train()
    dur = []
    for epoch in range(args.n_epochs):
        if epoch >= 3:
            t0 = time.time()
        group_logits = classify(embeddings, args.aggregate_mode)
        loss = loss_fcn(group_logits[train_mask], labels[train_mask])
        print(group_logits[:2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= 3:
            dur.append(time.time() - t0)
        _, indices = torch.max(group_logits, dim=1)
        TP = torch.sum(indices & labels)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TP:{:d} ".format(epoch, np.mean(dur), loss.item(),TP))
    # test the model
    test_nid = test_mask.nonzero().squeeze()
    acc, pre, recall, F1_score = evaluate(classify, embeddings, labels, test_nid, args.aggregate_mode)
    print("Test Accuracy {:.4f}   Test Precision {:.4f}   Test Recall {:.4f}  F1 score {:.4f}".format(acc, pre, recall, F1_score))
