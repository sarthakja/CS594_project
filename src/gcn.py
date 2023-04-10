import os
import pickle
<<<<<<< Updated upstream
=======
#from utils import convert_pygraph
import random

import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> Stashed changes

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

graphs_path = dir + "/data/graphs.pkl"

if os.path.isfile(graphs_path):
    with open(graphs_path, "rb") as handle:
        print("Retrieve the pickle!")
<<<<<<< Updated upstream
        positive_graphs, negative_graphs = pickle.load(handle)
=======
        dataset = pickle.load(handle)


NUM_CLASSES = 7
NUM_NODE_FEATURES = 33

torch.manual_seed(12345)
#shuffle the dataset
random.shuffle(dataset)

train_dataset = dataset[:1050]
test_dataset = dataset[1050:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

# a simple GIN model
class GIN(torch.nn.Module):
    # we need to design MLP for GINConv layer
    # first, try linear --> batch norm --> relu --> linear --> relu
    def __init__(self, hidden):
        super(GIN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GINConv(Sequential(Linear(NUM_NODE_FEATURES, hidden),
                                        BatchNorm1d(hidden), ReLU(),
                                        Linear(hidden, hidden), ReLU()))
        self.conv2 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden), ReLU(),
                                        Linear(hidden, hidden), ReLU()))
        self.conv3 = GINConv(Sequential(Linear(hidden, hidden),
                                        BatchNorm1d(hidden), ReLU(),
                                        Linear(hidden, hidden), ReLU()))
        self.lin1 = Linear(hidden*3, hidden*3)
        self.lin2 = Linear(hidden*3, NUM_CLASSES)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # 2. Readout layer
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        # 3. Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # 4. Apply a final classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h

gin = GIN(hidden=64)
gin = gin.to(torch.float64)


# a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(NUM_NODE_FEATURES, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = Linear(hidden, NUM_CLASSES)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


gcn = GCN(hidden=64)
gcn = gcn.to(torch.float64)

# set model

model = gin
def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(model, loader):
    model.eval()

    correct = 0
    y_pred_list = []
    y_label_list = []

    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        y_pred_list += list(pred.data.cpu().numpy())
        y_label_list += list(data.y.data.cpu().numpy())
    accuracy = correct / len(loader.dataset)
    return accuracy, y_pred_list, y_label_list


for epoch in range(100):
    train(model)
    train_acc, train_pred, train_label = test(model, train_loader)
    test_acc, test_pred, test_label = test(model, test_loader)

    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc}')
>>>>>>> Stashed changes

first_pos_graph = positive_graphs[0]
print("Number of nodes (1st graphs for positive label graph list: ")
print(first_pos_graph.number_of_nodes())


<<<<<<< Updated upstream
first_neg_graph = negative_graphs[0]
print("Number of nodes (1st graphs for negative label graph list: ")
print(first_neg_graph.number_of_nodes())
# TODO put the GCN class over here and metric methods
=======
# Build confusion matrix
cf_matrix = confusion_matrix(test_label, test_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sns.heatmap(df_cm, annot=True)
plt.savefig(dir + '/data/confusion_matrix.png')
>>>>>>> Stashed changes
