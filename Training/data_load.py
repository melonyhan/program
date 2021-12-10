# -*- coding: utf-8 -*-

import json
import torch
from networkx.readwrite import json_graph
import dgl
import random
from operator import itemgetter
import networkx as nx
import numpy as np


def data_load(network_name, label_rate, train_rate, Label_ratio):
    print("data_load")
    with open('../Data/networkx/{}-networkx.json'.format(network_name), 'r') as fr:
        G_data = json.load(fr)
    G = json_graph.node_link_graph(G_data)
    g = dgl.from_networkx(G)

    # 获取特征
    with open('../Data/TrainingFeatures/Betweenness/{}_betweenness.json'.format(network_name), 'r') as fr:  
        Betweenness = json.load(fr)
    with open('../Data/TrainingFeatures/Closeness/{}_closeness.json'.format(network_name), 'r') as fr:
        Closeness = json.load(fr)
    with open('../Data/TrainingFeatures/Cluster/{}_cluster.json'.format(network_name), 'r') as fr:
        Cluster = json.load(fr)
    with open('../Data/TrainingFeatures/Degree/{}_degree.json'.format(network_name), 'r') as fr:
        Degree = json.load(fr)
    with open('../Data/TrainingFeatures/Eigenvector/{}_eigenvector.json'.format(network_name), 'r') as fr:
        Eigenvector = json.load(fr)
    with open('../Data/TrainingFeatures/Kcore/{}_kcore.json'.format(network_name), 'r') as fr:
        Kcore = json.load(fr)
    with open('../Data/TrainingFeatures/LocalRank/{}_localrank.json'.format(network_name), 'r') as fr:
        LocalRank = json.load(fr)
    with open('../Data/TrainingFeatures/PageRank/{}_pagerank.json'.format(network_name), 'r') as fr:
        PageRank = json.load(fr)
    
    feat = [Betweenness, Closeness, Cluster, Degree, Eigenvector, Kcore, LocalRank,PageRank]
    g.ndata['feat'] = torch.tensor(feat).t()

    # 获取标签
    GroupAndLabel = {}
    with open('../Data/GroupInfluence/{}_GroupInfluence.json'.format(network_name), 'r') as fr:
        group_rank = json.load(fr)
    SortedGroup = sorted(group_rank,key=itemgetter(1),reverse = True)  # [[5, 71, 302, 841, 971]，476]
    label_num = round(len(group_rank) * label_rate)
    highgroup = []
    badgroup = []
    highgroup = [x[0] for x in SortedGroup[:label_num]]  # [5, 71, 302, 841, 971]
    badgroup = [x[0] for x in SortedGroup[-label_num * Label_ratio:]]
    label = []
    for singlegroupdata in group_rank:
        singlegroup = singlegroupdata[0]
        if singlegroup in highgroup:
            label.append(1)
        else:
            label.append(0)
    print("Get label")
    GroupAndLabel["label"] = torch.tensor(label)

    # 划分数据集
    random.shuffle(highgroup)
    random.shuffle(badgroup)
    n_train_highgroup = round(len(highgroup) * train_rate)
    n_train_badgroup = round(len(badgroup) * train_rate)    

    train_set = highgroup[:n_train_highgroup] + badgroup[:n_train_badgroup]
    test_set = highgroup[n_train_highgroup:] + badgroup[n_train_badgroup:]

    train_mask = [0] * len(group_rank)
    test_mask = [0] * len(group_rank)
    for i in range(0, len(group_rank)):
        if group_rank[i][0] in train_set:
            train_mask[i] = 1
        elif group_rank[i][0] in test_set:
            test_mask[i] = 1

    print("Get mask")
    GroupAndLabel["train_mask"] = torch.tensor(train_mask, dtype=torch.bool)
    GroupAndLabel["test_mask"] = torch.tensor(test_mask, dtype=torch.bool)
    GroupAndLabel["group_rank"] = group_rank
    # print(g.ndata['feat'])
    return g, GroupAndLabel


def GetMap(InputFile):
    rfile = open(InputFile)
    rfile.readline()
    rfile.readline()
    SumStr = rfile.readline()
    #print SumStr
    SumList = SumStr.split()
    #print SumList
    NodesNum = int(SumList[2])
    EdgesNum = int(SumList[4])
    #print("Total number of Nodes: %s" % NodesNum)
    #print("Total number of edges: %s" % EdgesNum)
    rfile.readline()
    NodeNum = 0    
    IDList = []
    MapDic = {}
    line = rfile.readline()
    while line:
        str2 = line[0:len(line)-1]
        #str1 = str2.split("\t")
        str1 = str2.split(" ")
        sourceID1 = int(str1[0])
        targetID1 = int(str1[1])
        if sourceID1 not in IDList:
            IDList.append(sourceID1)
        if targetID1 not in IDList:
            IDList.append(targetID1)
        '''if targetID!=sourceID:
            NetworkxDiGraph.add_edge(sourceID,targetID)
            print "Done with Edge: %s" % str2'''
        line = rfile.readline()
    IDList.sort()
    NodeNum = len(IDList)
    #print(IDList) 
    for NewID in range(0,len(IDList)):
        MapDic[IDList[NewID]] = NewID
    rfile.close()
    #print MapDic
    #print("GetMap and NodeNumber") 
    #Result = {}
    #Result["MapDic"] = MapDic
    #Result["NodeNum"] = NodeNum
    #print(MapDic) 
    return MapDic

def logits2Grouplogits(logits,group_rank,network_name):
    MapDic = GetMap('../Data/OriginalNetwork/{}.txt'.format(network_name))
    logits_list = logits.tolist()
    # print(np.array(logits_list).shape)  # [1005,128]
    #改造Logits和labels
    group_logits = []
    for i in range(0, len(group_rank)):
        group_feat_list = [logits_list[MapDic[k]] for k in group_rank[i][0]]  # [5,128]
        group_logits.append([sum(e)/len(e) for e in zip(*group_feat_list)])  #5个节点求平均
    #print(group_feat_list)
    #print(logits_list[0])
    #print(group_logits) 
    #print(labels[0])
    finallogits = torch.tensor(group_logits)
    # print(finallogits[0])
    finallogits.requires_grad_()
    return finallogits  # [15018,128]


# 构建图求卷积后，得到group的单个向量表示
def AggregateLogitsWithConvolution(network_name, group_list, node_logits):
    with open('../Data/networkx/{}-networkx.json'.format(network_name)) as fr:
        data = json.load(fr)
    g = json_graph.node_link_graph(data)
    node_list = sorted(g.nodes())
    group_out_list = []
    for group in group_list:
        group_logits = [node_logits[node_list.index(id),...] for id in group[0]]
        group_out = AggregateLogits(g, group[0], group_logits)
        group_out_list.append(group_out)
    group_out_list = torch.tensor(group_out_list)
    group_out_list.requires_grad_()
    return group_out_list
    

def AggregateLogits(g, group, logits):
    '''
    # 找出最接近网络中心节点的节点作为核节点
    with open('../Data/SortedCentrality/SortedFile-{}.json'.format(network_name), 'r') as fr:
        result = json.load(fr)
    core_id = result["SortedBetweennessNode"][0]
    core_distanace = nx.shortest_path_length(g,core_id,group[0])
    core = group[0]
    for i in range(1,len(group)):
        path_lenth = nx.shortest_path_length(g,core_id,group[i])
        if core_distanace > path_lenth:
            core_distanace = path_lenth
            core = group[i]
    '''
    edge_list = []
    distance_list = []

    g = g.to_undirected()
    # 判断图是否连通
    if nx.is_connected(g):
        for i in range(len(group)-1):
            for j in range(i+1, len(group)):
                path_lenth = nx.shortest_path_length(g, group[i], group[j])
                if path_lenth < nx.radius(g):
                    edge_list.append([group[i],group[j]])
                    distance_list.append(path_lenth)
    else:
        for i in range(len(group)-1):
            for j in range(i+1, len(group)):
                try:
                    path_lenth = nx.shortest_path_length(g, group[i], group[j])
                except nx.exception.NetworkXNoPath:
                    continue
                else:
                    edge_list.append([group[i],group[j]])
                    distance_list.append(path_lenth)
    new_G = nx.Graph()
    new_G.add_nodes_from(group)
    new_G.add_edges_from(edge_list)
    m = 0
    for [i, j] in edge_list:
        new_G[i][j]['weight'] = - 1 / ((distance_list[m] + 1) ** 2)
        m = m + 1

    A  = np.array(nx.adjacency_matrix(new_G).todense())
    E = np.mat(np.identity(len(group)))
    A = torch.tensor(A + E).to(torch.float32)
    logits = [t.detach().numpy() for t in logits]
    logits = torch.tensor(logits).to(torch.float32)
    convolution_result = torch.mm(A, logits) 

    logits_list = convolution_result.tolist()
    group_logits_out = [sum(e)/len(e) for e in zip(*logits_list)]

    return group_logits_out
