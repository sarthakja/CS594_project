import os
import pickle
import networkx as nx
import torch
import numpy as np
from random import shuffle
from torch.nn import Linear
import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import global_mean_pool

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

graphs_path = dir + "/data/graphs.pkl"

if os.path.isfile(graphs_path):
    with open(graphs_path, "rb") as handle:
        print("Retrieve the pickle!")
        positive_graphs, negative_graphs = pickle.load(handle)

first_pos_graph = positive_graphs[0]
print("Number of nodes (1st graphs for positive label graph list: ")
print(first_pos_graph.number_of_nodes())
print("Number of edges (1st graphs for positive label graph list: ")
print(first_pos_graph.number_of_edges())

print("Number of positive graphs: ", len(positive_graphs))
print("Number of negative graphs: ", len(negative_graphs))

first_neg_graph = negative_graphs[0]
print("Number of nodes (1st graphs for negative label graph list: ")
print(first_neg_graph.number_of_nodes())
print("Number of edges (1st graphs for negative label graph list: ")
print(first_neg_graph.number_of_edges())

# label pos & neg
for p in positive_graphs:
    p.graph.update({'y':1})

for n in negative_graphs:
    n.graph.update({'y':0})


# TODO put the GCN class over here and metric methods
combined = positive_graphs + negative_graphs
print(combined[0].graph) # first graph is positive
print(combined[24].graph) # last graph is negative

# Since we have 25 graphs total (12 positive, 13 negative), we can use the first 20 for training and 5 for testing
shuffle(combined)
train_graphs = combined[:20]
test_graphs = combined[20:]

print("Number of training graphs: ", len(train_graphs))
print("Number of testing graphs: ", len(test_graphs))

# Training the GNN
#print(list(train_graphs[0].edges()))

# converting each graph to adjacency matrix
adj_matrices = []
for graph in combined:
    adj_matrix = nx.to_numpy_matrix(graph)
    adj_matrices.append(adj_matrix)

#print(adj_matrices)

# getting num features
for i, j in enumerate(combined[0].nodes.data()):
    if i == 0:
        for node, dict in combined[0].nodes.data():
            num_node_features = len(dict)
print(num_node_features)

node_feature_matrices = []

for graph in combined:
    num_nodes = graph.number_of_nodes()
    node_features = np.zeros((num_nodes, num_node_features))

    for i, node in enumerate(graph.nodes()):
        # dictionary of node features
        node_dict = graph.nodes[node]
        #node_dict.pop('res_name')
        #node_dict.pop('res_ss')

        for j, (key, val) in enumerate(node_dict.items()):
            print(j, (key, val))
            node_features[i, j] = val

    node_feature_matrices.append(node_features)

print(node_feature_matrices)
