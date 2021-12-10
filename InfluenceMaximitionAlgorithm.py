#! /usr/bin/env python
#coding=utf-8
from __future__ import division
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import time
import random
import json
import os
import math
import networkx as nx
import ImModelMC as IC
import pandas as pd
import sys
from collections import Counter
from random import uniform, seed
from networkx.readwrite import json_graph
from operator import itemgetter
from networkx.algorithms import community
from heapdict import heapdict
import argparse


def greedy(NetworkxFilePath,ResultFilePath,k,p=0.1,mc=1000):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)      
    allnodes = nx.nodes(g)
    S, spread, timelapse, start_time = [], [], [], time.time()
    # Find k nodes with largest marginal gain
    for _ in range(k):
        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(allnodes)-set(S):
            # Get the spread
            s = IC.GroupInflunceBasedOnIC(g,S+[j],p,mc)
            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j
        # Add the selected node to the seed set
        S.append(node)
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
        print("%d nodes find" % len(S))
    #return(S,spread,timelapse)
    print(S)
    wfile = open(ResultFilePath,"w+")
    json.dump(S,wfile)
    rfile.close()
    wfile.close()
    print("Greedy done")
    

def celf(NetworkxFilePath,ResultFilePath,k,p,mc=1000):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    # Calculate the first iteration sorted list
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)
    allnodes = nx.nodes(g)
    start_time = time.time() 
    marg_gain = [IC.GroupInflunceBasedOnIC(g,[node],p,mc) for node in allnodes]
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(allnodes,marg_gain), key=lambda x: x[1],reverse=True)
    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [len(allnodes)], [time.time()-start_time]
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    for _ in range(k-1):    
        check, node_lookup = False, 0
        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1
            # Recalculate spread of top node
            current = Q[0][0]
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC.GroupInflunceBasedOnIC(g,S+[current],p,mc) - spread)
            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)
            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)
        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        # Remove the selected node from the list
        Q = Q[1:]
    #     print(len(S)) 
    # print(S)
    wfile = open(ResultFilePath,"w+")
    json.dump(S,wfile)
    rfile.close()
    wfile.close()    
    print("Celf done")
    #return(S,SPREAD,timelapse,LOOKUPS)


def degree_discount(NetworkxFilePath, ResultFilePath, n):
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)
    g = json_graph.node_link_graph(jsondata)
    edges = list(nx.edges(g))
    
    out_degree = {}
    connection = {}
    seeds = []

    for edge in edges:
        if edge[1] in connection:
            connection[edge[1]].append(edge[0])
        else:
            connection[edge[1]] = [edge[0]]

        if edge[0] in out_degree:
            out_degree[edge[0]] += 1
        else:
            out_degree[edge[0]] = 1

    while len(seeds) < n:
        seed = sorted(out_degree.items(), key=lambda item: item[1], reverse=True)[0][0]
        seeds.append(seed)

        out_degree[seed] = -1
        if seed in connection:
            for node in connection[seed]:
                out_degree[node] -= 1            
    # print(seeds)
    #return seeds
    wfile = open(ResultFilePath,"w+")
    json.dump(seeds,wfile)
    rfile.close()
    wfile.close()
    print("Degree discount done")


def degree_neighbor(NetworkxFilePath, ResultFilePath, n):
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)
    edges = list(nx.edges(g))    
    
    out_degree = {}
    centrality_score = {}
    seeds = []

    for edge in edges:
        if edge[0] in out_degree:
            out_degree[edge[0]] += 1
        else:
            out_degree[edge[0]] = 1

    centrality_score = out_degree.copy()

    for edge in edges:
        if edge[1] in out_degree:
            centrality_score[edge[0]] += out_degree[edge[1]]
        else:
            pass

    seeds = list({k: v for k, v in sorted(centrality_score.items(), key=lambda item: item[1], reverse=True)}.keys())[:n]
    # print(seeds)
    #return seeds
    wfile = open(ResultFilePath,"w+")
    json.dump(seeds,wfile)
    rfile.close()
    wfile.close()
    print("Degree neighbor done")


def mia(NetworkxFilePath, ResultFilePath, n, ta=0.1):
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)
    g.remove_edges_from(nx.selfloop_edges(g))
    edges = list(nx.edges(g))
    nodes = list(nx.nodes(g))
    
    out_connection = {}
    centrality_score = {}
    seeds = []

    for edge in edges:
        if edge[0] in out_connection:
            out_connection[edge[0]].append(edge[1])
        else:
            out_connection[edge[0]] = [edge[1]]

    for node in nodes:
        centrality_score[node] = mia_centrality(node, out_connection, 0, 1, ta)

    i = 0
    centrality_score = {k: v for k, v in sorted(centrality_score.items(), key=lambda item: item[0])}
    for node, _ in sorted(centrality_score.items(), key=lambda item: item[1], reverse=True):
        if i >= n:
            break
        else:
            i += 1
        seeds.append(node)
    # print(seeds)
    #return seeds
    wfile = open(ResultFilePath,"w+")
    json.dump(seeds,wfile)
    rfile.close()
    wfile.close()
    print("Mia done")
    

def mia_centrality(node, out_connection, centrality_score, ap, theta):
    #sys.setrecursionlimit(3000)
    if node not in out_connection.keys():
        return 1
    
    ap *= 1 / len(out_connection[node])
    if ap < theta:
        return 1
    
    for sub_node in out_connection[node]:
        centrality_score += mia_centrality(sub_node, out_connection, centrality_score, ap, theta)
    
    return centrality_score


def ris(NetworkxFilePath,ResultFilePath,k,p,mc=1000):    
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of RRSs to generate
    Return: A seed set of nodes as an approximate solution to the IM problem
    """
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)
    edges = list(nx.edges(g))
    source = [single[0] for single in edges]
    target = [single[1] for single in edges]
    G = pd.DataFrame({'source':source,'target':target})
    # Step 1. Generate the collection of random RRSs
    start_time = time.time()
    R = [get_RRS(G,p) for _ in range(mc)]
    # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
    SEED, timelapse = [], []
    for _ in range(k):
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(int(seed))
        # Remove RRSs containing last chosen seed 
        R = [rrs for rrs in R if seed not in rrs]
        # Record Time
        timelapse.append(time.time() - start_time)
    print(sorted(SEED))
    #return(sorted(SEED),timelapse)
    wfile = open(ResultFilePath,"w+")
    json.dump(sorted(SEED),wfile)
    rfile.close()
    wfile.close()
    print("Ris done")

def get_RRS(G,p):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: A random reverse reachable set expressed as a list of nodes
    """
    # Step 1. Select random source node
    source = random.choice(np.unique(G['source']))
    # Step 2. Get an instance of g from G by sampling edges  
    g = G.copy().loc[np.random.uniform(0,1,G.shape[0]) < p]
    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]   
    while new_nodes:
        # Limit to edges that flow into the source node
        temp = g.loc[g['target'].isin(new_nodes)]
        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()
        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))
        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))
        # Reset loop variables
        RRS0 = RRS[:]
    return(RRS)

class Node(object):
    def __init__(self, node):
        self.node = node
        self.mg1 = 0
        self.prev_best = None
        self.mg2 = 0
        self.flag = None
        self.list_index = 0

def celfpp(NetworkxFilePath, ResultFilePath, k, p, mc=1000):
    rfile = open(NetworkxFilePath)
    jsondata = json.load(rfile)    
    g = json_graph.node_link_graph(jsondata)
    g.remove_edges_from(nx.selfloop_edges(g))
    #edges = list(nx.edges(g))
    nodes = list(nx.nodes(g))    
    S = set()
    # Note that heapdict is min heap and hence add negative priorities for
    # it to work.
    Q = heapdict()
    last_seed = None
    cur_best = None
    node_data_list = []
    #IC.GroupInflunceBasedOnIC(g,[node],p,mc)
    for node in nodes:
        node_data = Node(node)
        node_data.mg1 = IC.GroupInflunceBasedOnIC(g,[node],p,mc)
        node_data.prev_best = cur_best
        node_data.mg2 = IC.GroupInflunceBasedOnIC(g,[node, cur_best.node],p,mc) if cur_best else node_data.mg1
        node_data.flag = 0
        cur_best = cur_best if cur_best and cur_best.mg1 > node_data.mg1 else node_data
        #graph.nodes[node]['node_data'] = node_data
        node_data_list.append(node_data)
        node_data.list_index = len(node_data_list) - 1
        Q[node_data.list_index] = - node_data.mg1
    while len(S) < k:
        node_idx, _ = Q.peekitem()
        node_data = node_data_list[node_idx]
        if node_data.flag == len(S):
            S.add(node_data.node)
            del Q[node_idx]
            last_seed = node_data
            continue
        elif node_data.prev_best == last_seed:
            node_data.mg1 = node_data.mg2
        else:
            before = IC.GroupInflunceBasedOnIC(g,list(S),p,mc)
            S.add(node_data.node)
            after = IC.GroupInflunceBasedOnIC(g,list(S),p,mc)
            S.remove(node_data.node)
            node_data.mg1 = after - before
            node_data.prev_best = cur_best
            S.add(cur_best.node)
            before = IC.GroupInflunceBasedOnIC(g,list(S),p,mc)
            S.add(node_data.node)
            after = IC.GroupInflunceBasedOnIC(g,list(S),p,mc)
            S.remove(cur_best.node)
            if node_data.node != cur_best.node: S.remove(node_data.node)
            node_data.mg2 = after - before
        if cur_best and cur_best.mg1 < node_data.mg1:
            cur_best = node_data
        node_data.flag = len(S)
        Q[node_idx] = - node_data.mg1
    #     print(len(S))
    # print(S)
    #return S
    wfile = open(ResultFilePath,"w+")
    json.dump(list(S),wfile)
    rfile.close()
    wfile.close()
    print("Celfpp done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use influence maximition algorithm')
    parser.add_argument("network_name", type=str)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--ta", type=float, default=0.1)
    args = parser.parse_args()

    greedy("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-greedy-{}-{}-{}.json".format(args.network_name,args.size,args.network_name,args.size,args.p,args.mc),
            args.size,
            args.p,
            args.mc)
    
    celf("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-celf-{}-{}-{}.json".format(args.network_name,args.size,args.network_name,args.size,args.p,args.mc),
            args.size,
            args.p,
            args.mc)

    celfpp("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-celfpp-{}-{}-{}.json".format(args.network_name,args.size,args.network_name,args.size,args.p,args.mc),
            args.size,
            args.p,
            args.mc)

    mia("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-mia-{}-{}.json".format(args.network_name,args.size,args.network_name,args.size,args.ta),
            args.size,
            args.ta)

    ris("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-ris-{}-{}-{}.json".format(args.network_name,args.size,args.network_name,args.size,args.p,args.mc),
            args.size,
            args.p,
            args.mc)

    degree_discount("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-degreediscount-{}.json".format(args.network_name,args.size,args.network_name,args.size),
            args.size)

    degree_neighbor("Data/networkx/{}-networkx.json".format(args.network_name),
            "Data/InfluenceMaximizationGroup/{}/{}/{}-degreeneighbor-{}.json".format(args.network_name,args.size,args.network_name,args.size),
            args.size)

