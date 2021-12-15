
import argparse
from gensim.models import Word2Vec

from data_load import *
p = 1
q = 2


def gen_graph():
    g = nx.Graph()
    g = nx.DiGraph()
    g.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 1.5), (4, 1, 1.0), (2, 4, 0.5), (4, 5, 1.0)])
    g.add_weighted_edges_from([(2, 1, 0.5), (3, 2, 1.5), (1, 4, 1.0), (4, 2, 0.5), (5, 4, 1.0)])
    return g


def get_alias_edge(g, prev, cur):
    unnormalized_probs = []
    for cur_nbr in g.neighbors(cur):
        if cur_nbr == prev:
            unnormalized_probs.append(g[cur][cur_nbr]['weight']/p)
        elif g.has_edge(cur_nbr, prev):
            unnormalized_probs.append(g[cur][cur_nbr]['weight'])
        else:
            unnormalized_probs.append(g[cur][cur_nbr]['weight']/q)
    norm = sum(unnormalized_probs)
    normalized_probs = [float(prob)/norm for prob in unnormalized_probs]
    return alias_setup(normalized_probs)


def alias_setup(ws):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(ws)
    probs = np.zeros(K, dtype=np.float32)
    alias = np.zeros(K, dtype=np.int32)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        probs[kk] = K*prob
        if probs[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        alias[small] = large
        probs[large] = probs[large] + probs[small] - 1.0
        if probs[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return alias, probs


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


def alias_draw(alias, probs):
    num = len(alias)
    k = int(np.floor(np.random.rand() * num))
    if np.random.rand() < probs[k]:
        return k
    else:
        return alias[k]


def preprocess_transition_probs(g):
    '''
    Preprocessing of transition probabilities for guiding the random walks.
    '''
    alias_nodes = {}
    for node in g.nodes():
        unnormalized_probs = [g[node][nbr]['weight']
                              for nbr in g.neighbors(node)]
        norm= sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm for u_prob in unnormalized_probs]
        alias_nodes[node] = alias_setup(normalized_probs)

    alias_edges = {}
    for edge in g.edges():
        alias_edges[edge] = get_alias_edge(g, edge[0], edge[1])
    return alias_nodes, alias_edges


def node2vec_walk(g, walk_length, start_node, alias_nodes, alias_edges):
    '''
    Simulate a random walk starting from start node.
    '''
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(g.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(
                    cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
            else:
                prev = walk[-2]
                pos = (prev, cur)
                next = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                walk.append(next)
        else:
            break
    return walk


def simulate_walks(g, num_walks, walk_length, alias_nodes, alias_edges):
    '''
    Repeatedly simulate random walks from each node.
    '''
    walks = []
    nodes = list(g.nodes())
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print("iteration: {} / {}".format(walk_iter + 1, num_walks))
        random.shuffle(nodes)
        for node in nodes:
            walks.append(node2vec_walk(g, walk_length=walk_length, start_node=node, alias_nodes=alias_nodes, alias_edges=alias_edges))
    return walks


# def learn_embeddings(walks):
#     '''
#     Learn embeddings by optimizing the Skipgram objective using SGD.
#     '''
#     walks = [map(str, walk) for walk in walks]
#     model = Word2Vec(walks, 128, window=10, min_count=0, sg=1, workers=8)
#     # model.save_word2vec_format(args.output)
#     return


def get_embeddings(w2v_model, graph):
    count = 0
    invalid_word = []
    _embeddings = {}
    for word in graph.nodes():
        if word in w2v_model.wv:
            _embeddings[word] = w2v_model.wv[word]
        else:
            invalid_word.append(word)
            count += 1
    print("无效word", len(invalid_word))
    print("有效embedding", len(_embeddings))
    # print(_embeddings)
    return _embeddings


def main(args):
    print("main begin")
    with open('../Data/networkx/{}-networkx.json'.format(args.network_name), 'r') as fr:
        G_data = json.load(fr)
    G = json_graph.node_link_graph(G_data)
    # g = dgl.from_networkx(G)
    g = nx.Graph(G)
    print(g)
    # g, GroupAndLabel = data_load(args.network_name, args.label_rate, args.train_rate, args.label_ratio)
    alias_nodes, alias_edges = preprocess_transition_probs(g)
    walks = simulate_walks(g, 1, 3, alias_nodes, alias_edges)

    # print(walks)
    # kwargs = {"sentences": walks, "min_count": 0, "vector_size": 128,
    #           "sg": 1, "hs": 0, "workers": 3, "window": 5, "epochs": 3}
    # model = Word2Vec(**kwargs)
    # embeddings = get_embeddings(model, g)
# if __name__ == '__main__':
#     g = gen_graph()
#     # args = parser.parse_args()
#     # g, GroupAndLabel = data_load(args.network_name, args.label_rate, args.train_rate, args.label_ratio)
#
#     print(g)
#     alias_nodes, alias_edges = preprocess_transition_probs(g)
#     walks = simulate_walks(g, 2, 3, alias_nodes, alias_edges)
#
#     # w2v_model = Word2Vec(walks, 128, window=10, min_count=0, sg=1, workers=8)
#     print(walks)
#     kwargs = {"sentences": walks, "min_count": 0, "vector_size": 64,
#               "sg": 1, "hs": 0, "workers": 3, "window": 5, "epochs": 3}
#     model = Word2Vec(**kwargs)
#     embeddings = get_embeddings(model, g)
#     # print(w2v_model)
#     # Walk iteration:
#     # iteration: 1 / 2
#     # iteration: 2 / 2
#     # [[5, 4, 1], [2, 3, 2], [4, 1, 2], [3, 2, 3], [1, 2, 3], [4, 1, 2], [3, 2, 3], [1, 2, 3], [2, 3, 2], [5, 4, 1]]
#
#
