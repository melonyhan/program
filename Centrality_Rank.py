
import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
from operator import itemgetter
import argparse

''' Use this file to compute centrality features, and rank the nodes with the centrality result. 
    Use 'python Centrality_Rank.py + network name' to execute this file.'''

# compute the Local Rank value of nodes
def Local_Rank(G):
    R_list = {}
    for i in sorted(list(G.nodes)):
        H = [list(G[id]) for id in list(G[i])]
        neignbor_2 = []
        for l in list(H):
            neignbor_2.extend(l)
        neignbor_2.extend(list(G[i]))
        num_neignbor_2 = len(set(neignbor_2)) - 1
        R_list[i] = num_neignbor_2
    Q_list = {}
    for i in sorted(list(G.nodes)):
        Q = 0
        for id in G[i]:
            Q = Q + R_list[id]
        Q_list[i] = Q
    LR_list = []
    LR_array = []
    for i in sorted(list(G.nodes)):
        LR = 0
        for id in G[i]:
            LR = LR + Q_list[id]
        LR_list.append([i, LR])
        LR_array.append(LR)
    sortedlist = sorted(LR_list, key=itemgetter(1), reverse = True)  
    sorted_node_list = [x[0] for x in sortedlist]
    LocalRank_list = list(np.true_divide(np.array(LR_array) - min(LR_array), max(LR_array) - min(LR_array)))

    return sorted_node_list, LocalRank_list

def SortedNodesAccording2Centrality(network_name):
    result = {}
    rfile = open('./Data/networkx/{}-networkx.json'.format(network_name))
    jsondata = json.load(rfile)    
    NetworkxGraph = json_graph.node_link_graph(jsondata)
    NetworkxGraph.remove_edges_from(nx.selfloop_edges(NetworkxGraph))
    #degree
    degreedic = nx.degree_centrality(NetworkxGraph)
    degreelist = [(key,val) for key,val in degreedic.items()]
    sorteddegreelist = sorted(degreelist,key=itemgetter(1),reverse = True)  
    result["SortedDegreeNode"] = [x[0] for x in sorteddegreelist]
    degree_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        degree_array.append(degreedic[int(i)])
    degree_array = list(np.true_divide(np.array(degree_array) - min(degree_array), max(degree_array) - min(degree_array)))  
    result["SortedDegreeList"] = sorted(degree_array)
    with open('./Data/TrainingFeatures/Degree/{}_degree.json'.format(network_name), 'w') as fw:
        json.dump(degree_array, fw)
    print("Done with degree")
    
    #Local Rank
    result["SortedLocalRankNode"], LR_array = Local_Rank(NetworkxGraph)
    result["SortedLocalRankList"] = sorted(LR_array)
    with open('./Data/TrainingFeatures/LocalRank/{}_localrank.json'.format(network_name), 'w') as fw:
        json.dump(LR_array, fw)
    print("Done with local rank")

    #eigenvector
    eigenvectordic = nx.eigenvector_centrality(NetworkxGraph)
    eigenvectorlist = [(key,val) for key,val in eigenvectordic.items()]
    sortedeigenvectorlist = sorted(eigenvectorlist,key=itemgetter(1),reverse = True)  
    result["SortedEigenvectorNode"] = [x[0] for x in sortedeigenvectorlist]
    eigenvector_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        eigenvector_array.append(eigenvectordic[int(i)])
    eigenvector_array = list(np.true_divide(np.array(eigenvector_array) - min(eigenvector_array), max(eigenvector_array) - min(eigenvector_array)))  
    result["SortedEigenvectorList"] = sorted(eigenvector_array)
    with open('./Data/TrainingFeatures/Eigenvector/{}_eigenvector.json'.format(network_name), 'w') as fw:
        json.dump(eigenvector_array, fw)
    print("Done with eigenvector") 

    #clustering
    clusterdic = nx.clustering(NetworkxGraph)
    clusterlist = [(key,val) for key,val in clusterdic.items()]
    sortedclusterlist = sorted(clusterlist,key=itemgetter(1),reverse = True)  
    result["SortedClusterNode"] = [x[0] for x in sortedclusterlist]
    cluster_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        cluster_array.append(clusterdic[int(i)])
    cluster_array = list(np.true_divide(np.array(cluster_array) - min(cluster_array), max(cluster_array) - min(cluster_array)))  
    result["SortedClusterList"] = sorted(cluster_array)
    with open('./Data/TrainingFeatures/Cluster/{}_cluster.json'.format(network_name), 'w') as fw:
        json.dump(cluster_array, fw)
    print("Done with clustering") 

    #k-core
    kcoredic = nx.core_number(NetworkxGraph)
    kcorelist = [(key,val) for key,val in kcoredic.items()]
    sortedkcorelist = sorted(kcorelist,key=itemgetter(1),reverse = True)
    result["SortedKcoreNode"] = [x[0] for x in sortedkcorelist]
    kcore_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        kcore_array.append(kcoredic[int(i)])
    kcore_array = list(np.true_divide(np.array(kcore_array) - min(kcore_array), max(kcore_array) - min(kcore_array)))  
    result["SortedKcoreList"] = sorted(kcore_array)
    with open('./Data/TrainingFeatures/Kcore/{}_kcore.json'.format(network_name), 'w') as fw:
        json.dump(kcore_array, fw)
    print("Done with k-core") 

    #pagerank
    pagerankdic = nx.pagerank(NetworkxGraph)
    pageranklist = [(key,val) for key,val in pagerankdic.items()]
    sortedpageranklist = sorted(pageranklist,key=itemgetter(1),reverse = True) 
    result["SortedPagerankNode"] = [x[0] for x in sortedpageranklist]
    pagerank_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        pagerank_array.append(pagerankdic[int(i)])
    pagerank_array = list(np.true_divide(np.array(pagerank_array) - min(pagerank_array), max(pagerank_array) - min(pagerank_array)))  
    result["SortedPagerankList"] = sorted(pagerank_array)
    with open('./Data/TrainingFeatures/PageRank/{}_pagerank.json'.format(network_name), 'w') as fw:
        json.dump(pagerank_array, fw)
    print("Done with pagerank") 

    #closeness
    closenessdic = nx.closeness_centrality(NetworkxGraph)
    closenesslist = [(key,val) for key,val in closenessdic.items()]
    sortedcloselist = sorted(closenesslist,key=itemgetter(1),reverse = True) 
    result["SortedClosenessNode"] = [x[0] for x in sortedcloselist]
    closeness_array = []
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        closeness_array.append(closenessdic[int(i)])
    closeness_array = list(np.true_divide(np.array(closeness_array) - min(closeness_array), max(closeness_array) - min(closeness_array)))  
    result["SortedClosenessList"] = sorted(closeness_array)
    with open('./Data/TrainingFeatures/Closeness/{}_closeness.json'.format(network_name), 'w') as fw:
        json.dump(closeness_array, fw)
    print("Done with closeness")

    #betweenness
    betwdic = nx.betweenness_centrality(NetworkxGraph)
    betwlist = [(key,val) for key,val in betwdic.items()]
    sortedbetwlist = sorted(betwlist,key=itemgetter(1),reverse = True)
    result["SortedBetweennessNode"] = [x[0] for x in sortedbetwlist]
    Betweenness_array = []  
    for i in list(map(str, sorted(list(map(int ,NetworkxGraph.nodes()))))):
        Betweenness_array.append(betwdic[int(i)])
    Betweenness_array = list(np.true_divide(np.array(Betweenness_array) - min(Betweenness_array), max(Betweenness_array) - min(Betweenness_array)))  
    result["SortedBetweennessList"] = sorted(Betweenness_array)
    with open('./Data/TrainingFeatures/Betweenness/{}_betweenness.json'.format(network_name), 'w') as fw:
        json.dump(Betweenness_array, fw)
    print("Done with betweenness")
    
    # save the result
    with open('./Data/SortedCentrality/SortedFile-{}.json'.format(network_name), 'w') as fw:
        json.dump(result, fw)
    print("Done with sorting nodes according their centralities")
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get features and ranked centralities')
    parser.add_argument("network_name", type=str)
    args = parser.parse_args()

    SortedNodesAccording2Centrality(args.network_name)