#! /usr/bin/env python
#coding=utf-8
import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import argparse

def GroupInflunceBasedOnIC(g, S, p, mc):#p为节点间相互激活的概率
    """
    input
    g: the graph file of networkxversion
    S: seed set
    p: Activation limits
    mc: Number of cycles of Monte Carlo
    output
    the Influence of current seed set
    """
    '''networkfile = open(NetworkxFilePath)
    jsondata = json.load(networkfile)    
    g = json_graph.node_link_graph(jsondata)'''    
    spread = []
    for i in range(mc):
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                #np.random.seed(i)
                success = np.random.uniform(0, 1, len(list(nx.neighbors(g,node)))) < p
                new_ones += list(np.extract(success, list(nx.neighbors(g,node))))
            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread.append(len(A))
        #print("Done with %s cycle" % (i+1))
    #print(spread)
    #print(np.mean(spread))
    return round(np.mean(spread))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get group influence with IC')
    parser.add_argument("network_name", type=str)
    parser.add_argument("--p", type=float, default='0.05')
    parser.add_argument("--mc", type=int, default='100')
    args = parser.parse_args()

    with open('./Data/GroupData/{}-groups.json'.format(args.network_name), 'r') as fr:
        group_data = json.load(fr)
    
    with open('./Data/networkx/{}-networkx.json'.format(args.network_name), 'r') as fr:
        jsondata = json.load(fr)    
    g = json_graph.node_link_graph(jsondata)

    result = []
    i = 0
    for group in group_data:
        IC_influence = GroupInflunceBasedOnIC(g, group, args.p, args.mc)
        result.append([group, IC_influence])
        i = i + 1
        print(i,'/',len(group_data))
    
    with open('./Data/GroupInfluence/{}_GroupInfluence.json'.format(args.network_name), 'w') as fw:
        json.dump(result, fw)
