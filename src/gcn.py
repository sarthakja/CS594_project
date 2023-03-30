import os
import pickle

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


first_neg_graph = negative_graphs[0]
print("Number of nodes (1st graphs for negative label graph list: ")
print(first_neg_graph.number_of_nodes())
# TODO put the GCN class over here and metric methods
