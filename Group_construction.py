#! /usr/bin/env python
#coding=utf-8
import random
import json
import os
import networkx as nx
from networkx.readwrite import json_graph
import argparse

'''Use this file to construct group data'''

######################################### part 1 of group construction #########################################

# 变异过程函数
def GenerateNodeGroupFromMutation(NetworkxFilePath, InitialGroup, MutationTime):
    #对所有的节点组列表进行sorted，然后判断是否在最终的节点组列表中
    rfile1 = open(NetworkxFilePath)
    jsondata = json.load(rfile1)    
    G = json_graph.node_link_graph(jsondata)
    NodeList = sorted(nx.nodes(G))

    OriginalGroupSet = []
    for singleGroup in InitialGroup:
        if sorted(singleGroup) not in OriginalGroupSet:
            OriginalGroupSet.append(sorted(singleGroup))
    FinalGroupSet = OriginalGroupSet[:]
    print(len(FinalGroupSet))
    for singleSimple in OriginalGroupSet:
        for i in range(len(singleSimple)):
            count = 0
            while count < MutationTime:
                randnum = random.randint(0,(len(NodeList)-1))
                if NodeList[randnum] not in singleSimple:
                    copySimple = singleSimple[:]
                    copySimple[i] = NodeList[randnum]
                    sortedcopy = sorted(copySimple)
                    if sortedcopy not in FinalGroupSet:
                        FinalGroupSet.append(sortedcopy)
                        count+=1
        print("Done with group:%s" % singleSimple)
        print(len(FinalGroupSet))
    print("length of group:%d" % len(FinalGroupSet))
    return FinalGroupSet


# 将三种策略得到的最优节点组组成集合，并统一进行变异操作，输出为扩张后的节点组集合
def CombineAndMutation(network_name, group_size, MutationTime):
    Group_list = []
    # Get centrality result
    with open('./Data/SortedCentrality/SortedFile-{}.json'.format(network_name), 'r') as fr:
        centralityData = json.load(fr)
    Group_list.append(centralityData["SortedDegreeNode"][0:group_size])
    Group_list.append(centralityData["SortedEigenvectorNode"][0:group_size])
    Group_list.append(centralityData["SortedClusterNode"][0:group_size])
    Group_list.append(centralityData["SortedKcoreNode"][0:group_size])
    Group_list.append(centralityData["SortedPagerankNode"][0:group_size])
    Group_list.append(centralityData["SortedClosenessNode"][0:group_size])
    Group_list.append(centralityData["SortedLocalRankNode"][0:group_size])
    Group_list.append(centralityData["SortedBetweennessNode"][0:group_size])
    # add negative samples
    Group_list.append(centralityData["SortedDegreeNode"][-group_size:])
    Group_list.append(centralityData["SortedEigenvectorNode"][-group_size:])
    Group_list.append(centralityData["SortedClusterNode"][-group_size:])
    Group_list.append(centralityData["SortedKcoreNode"][-group_size:])
    Group_list.append(centralityData["SortedPagerankNode"][-group_size:])
    Group_list.append(centralityData["SortedClosenessNode"][-group_size:])
    Group_list.append(centralityData["SortedLocalRankNode"][-group_size:])
    Group_list.append(centralityData["SortedBetweennessNode"][-group_size:])
    # Get IC influence result
    with open('./Data/SortedSingleInfluence/{}-ICinfluence.json'.format(network_name), 'r') as fr:
        SortedSingleInfluence = json.load(fr)
    Group_list.append(SortedSingleInfluence["SortedInfluenceNode"][0:group_size])
    Group_list.append(SortedSingleInfluence["SortedInfluenceNode"][-group_size:])
    # Get influence maximition algorithm result
    for root, dirs, files in os.walk('./Data/InfluenceMaximizationGroup/{}/{}'.format(network_name,args.size)):
        for file in files:
            with open('{}/{}'.format(root,file)) as fr:
                data = json.load(fr)
                Group_list.append(data)
            print("Done with file:%s" % file)
    
    NetworkxFilePath = './Data/networkx/{}-networkx.json'.format(network_name)
    GroupData = GenerateNodeGroupFromMutation(NetworkxFilePath, Group_list, MutationTime)
    print("part 1 finished")
    print("length of group:%d" % len(GroupData))
    return GroupData


######################################### part 2 of group construction #########################################

def GenerateNodeGroupFromTopNRandom(network_name, GroupData, group_size, top_N, GroupNum):
    #generate initial node set
    InitialNodeSet = set()

    with open('./Data/SortedCentrality/SortedFile-{}.json'.format(network_name), 'r') as fr:
        centralityData = json.load(fr)
    for single in centralityData["SortedDegreeNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedEigenvectorNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedClusterNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedKcoreNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedPagerankNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedClosenessNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedLocalRankNode"][0:top_N]:
        InitialNodeSet.add(single)
    for single in centralityData["SortedBetweennessNode"][0:top_N]:
        InitialNodeSet.add(single)

    with open('./Data/SortedSingleInfluence/{}-ICinfluence.json'.format(network_name), 'r') as fr:
        SortedSingleInfluence = json.load(fr)

    for single in SortedSingleInfluence["SortedInfluenceNode"][0:top_N]:
        InitialNodeSet.add(single)

    InitialNodeList = list(InitialNodeSet)
    count = 0
    while count < GroupNum:
        randomgroup = sorted(random.sample(InitialNodeList, group_size))
        if randomgroup not in GroupData:
            GroupData.append(randomgroup)
            count+=1
            # print(count)
    print("part 2 finished")
    print("length of group:%d" % len(GroupData))
    return GroupData

######################################### part 3 of group construction #########################################

def GenerateNodeGroupFromAllNodesRandom(network_name, GroupData, group_size, GroupNum):
    with open('./Data/networkx/{}-networkx.json'.format(network_name)) as fr:
        jsondata = json.load(fr)      
    G = json_graph.node_link_graph(jsondata)
    NodeList = sorted(nx.nodes(G))

    count = 0
    while count < GroupNum:
        randomgroup = sorted(random.sample(NodeList, group_size))
        if randomgroup not in GroupData:
            GroupData.append(randomgroup)
            count+=1
            # print(count)
    print("part 3 finished")
    print("length of group:%d" % len(GroupData))
    return GroupData
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='construct different groups')
    parser.add_argument("network_name", type=str)
    parser.add_argument("--size", type=int, default=5, help='size of groups')
    parser.add_argument("--times", type=int, default=100, help='mutation times')
    parser.add_argument("--topN", type=int, default=50, help='choose the top N nodes to construct a node set in part 2')
    parser.add_argument("--num1", type=int, default=1000, help='the number of groups in part 2')
    parser.add_argument("--num2", type=int, default=5000, help='the number of groups in part 3')
    args = parser.parse_args()

    Group_data = CombineAndMutation(args.network_name, args.size, args.times)
    Group_data = GenerateNodeGroupFromTopNRandom(args.network_name, Group_data, args.size, args.topN, args.num1)
    Group_data = GenerateNodeGroupFromAllNodesRandom(args.network_name, Group_data, args.size, args.num2)

    with open('./Data/GroupData/{}-groups.json'.format(args.network_name), 'w') as fw:
        json.dump(Group_data, fw)
