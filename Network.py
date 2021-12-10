from __future__ import division
import json
import networkx as nx
from networkx.readwrite import json_graph
import argparse


def generateNetworkxGraphFromStanford(OriginalFilePath,NetworkxFilePath):
    rfile = open(OriginalFilePath)
    print(rfile.readline())
    print(rfile.readline())
    '''datalist = rfile.readlines()
    length = len(datalist)'''
    #print length
    SumStr = rfile.readline()
    #print SumStr
    SumList = SumStr.split()
    #print SumList
    NodesNum = int(SumList[2])
    EdgesNum = int(SumList[4])
    print("Total number of Nodes: %s" % NodesNum)
    print("Total number of edges: %s" % EdgesNum)
    print(rfile.readline())
    NetworkxGraph = nx.DiGraph()
    n=0
    line = rfile.readline()
    while line:
        n+=1
        str2 = line[0:len(line)-1]
        #print str2
        #print len(str2)
        str1 = str2.split()
        #print str1[0]
        sourceID = int(str1[0])
        targetID = int(str1[1])
        #print sourceID
        #print targetID
        NetworkxGraph.add_edge(sourceID,targetID)
        line = rfile.readline()
    print("Add edges done")
    wfile = open(NetworkxFilePath,"w+")
    GraphJsonData = json_graph.node_link_data(NetworkxGraph)
    json.dump(GraphJsonData,wfile)
    rfile.close()
    wfile.close()
    print("Done with generating networkx graph") 
    
    


def CreatNetworksGFromExistingFDirectedUnWeighted(OwnFilePath,outputfile):
    rfile = open(OwnFilePath)
    Jsondata = json.load(rfile)
    NodeList = Jsondata["NodeSet"]
    NetworkxGraph = nx.DiGraph()
    NodesNum = len(NodeList)
    for i in range(0,NodesNum+1):
        NetworkxGraph.add_node(i)
    print("Add nodes done")
    for SingleNode in NodeList:
        OutList = SingleNode["Step1OutNei"]
        #NeighborList = SingleNode["Neighbors"]
        for SingleOut in OutList:
            NetworkxGraph.add_edge(SingleNode["ID"],SingleOut)
        print("Done with Node: %s" % SingleNode["ID"]) 
    wfile = open(outputfile,"w+")
    GraphJsonData = json_graph.node_link_data(NetworkxGraph)
    json.dump(GraphJsonData,wfile)
    rfile.close()
    wfile.close()
    print("Done with generating networkx graph")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graph')

    parser.add_argument("network_name", type=str,
                        help="get network name")
    args = parser.parse_args()
    
    generateNetworkxGraphFromStanford("Data/OriginalNetwork/{}.txt".format(args.network_name),"Data/networkx/{}-networkx.json".format(args.network_name))