import os
import pickle
from utils import convert_pygraph
from torch_geometric.loader import DataLoader

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

networkx_graphs_path = dir + "/data/graphs_040423.pkl"

if os.path.isfile(networkx_graphs_path):
    with open(networkx_graphs_path, "rb") as handle:
        print("Retrieve the pickle!")
        graphs, labels = pickle.load(handle)

dataset = convert_pygraph(graphs[:20], labels[:20])
loader = DataLoader(dataset, batch_size=10)

for batch in loader:
    print(batch)
    print(batch.num_graphs)