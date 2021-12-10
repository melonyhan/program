#! /usr/bin/env python
#coding=utf-8
import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
from operator import itemgetter
import argparse

''' Use this file to compute influence of single node based on IC. 
    Use 'python Get_single_IC_influence.py + network name' to execute this file.
'''

def SingleInflunceBasedOnIC(network_name, p, mc):
    rfile = open('./Data/networkx/{}-networkx.json'.format(network_name))
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)    
    result = {}
    InfluenceList = []
    ThisNodeList = list(nx.nodes(g))
    for singlenode in ThisNodeList:
        S = []
        S.append(singlenode)
        nodeinfluence=[]
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
            nodeinfluence.append(len(A))
            #print("Done with %s cycle" % (i+1))
        InfluenceList.append([singlenode, np.mean(nodeinfluence)])
        print("Done with Monte Carlo Simulation for node %d" % singlenode)
    sortedinfluencelist = sorted(InfluenceList,key=itemgetter(1),reverse = True)
    result["SortedInfluenceList"] = [x[1] for x in sortedinfluencelist]
    result["SortedInfluenceNode"] = [int(x[0]) for x in sortedinfluencelist]

    with open('./Data/SortedSingleInfluence/{}-ICinfluence.json'.format(network_name), 'w') as fw:
        json.dump(result, fw)
    print("Done with sorting nodes according to Monte Carlo Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get features and ranked centralities')
    parser.add_argument("network_name", type=str)
    parser.add_argument("--p", type=float, default='0.05')
    parser.add_argument("--mc", type=int, default='100')
    args = parser.parse_args()

    SingleInflunceBasedOnIC(args.network_name, args.p, args.mc)