import os
import pickle
import random

import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date = "042523"
num_epoches = 2000
graph_CM = False
graph_F1 = True


networkx_graphs_path = dir + "/data/torch_graphs_041723_v2.pkl"
macro_path = dir + f"/data/GCN_macro_{date}_ep{num_epoches}.txt"

if graph_CM:
    conf_matrix_path = dir + f'/data/CM_GCN_{date}.png'

if graph_F1:
    train_test_f1_path = dir + f"/data/f1_{date}_ep{num_epoches}.png"

if os.path.isfile(networkx_graphs_path):
    with open(networkx_graphs_path, "rb") as handle:
        print("Retrieve the pickle!")
        dataset = pickle.load(handle)


NUM_CLASSES = 6
NUM_NODE_FEATURES = 33

torch.manual_seed(12345)

#shuffle the dataset
random.shuffle(dataset)

print(f"NUmber of graphs in the datasets: {len(dataset)}")


#Class label for each graph in labels
labels = np.empty(len(dataset))
for i in range(len(dataset)):
  labels[i] = dataset[i].y.item()

#labelCount stores the no of graphs for each class
labelCount = np.zeros(NUM_CLASSES)

#labelCount = {}
for i in range(len(labels)):

  labelCount[int(labels[i])]+=1

print(labelCount)

maxClassGraphs = np.amax(labelCount)

#Calculating class weights
classWeights = maxClassGraphs/labelCount
classWeights = torch.from_numpy(classWeights)
print(classWeights)

train_dataset , test_dataset = train_test_split(dataset, test_size=0.2, stratify=labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

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


model = GCN(hidden=64)
model = model.to(torch.float64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()

classWeights = classWeights.to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)

def train():
    model.train()

    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(loader):
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
    f1Score = f1_score(y_label_list, y_pred_list, average='weighted')
    return accuracy, f1Score, y_pred_list, y_label_list

test_f1 = []
train_f1 = []
epoches = []
test_preds = []
test_labels = []

for epoch in range(1, num_epoches):

    train()
    train_acc, train_f1Score, train_pred, train_label = test(train_loader)
    test_acc, test_f1Score, test_pred, test_label = test(test_loader)

    test_labels += list(test_label)
    test_preds += list(test_pred)

    if epoch % 20 == 0:
        epoches.append(epoch)
        train_f1.append(train_f1Score)
        test_f1.append(test_f1Score)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc}, Train F1: {train_f1Score: .4f}, '
              f'Test F1: {test_f1Score:.4f}')



with open(macro_path, "w") as f:
    f.write("Macro F1 for GCN model\n")
    # Print the precision and recall, among other metrics for testing set
    macro = classification_report(test_labels, test_preds, digits=3)
    print(macro)
    f.write(macro)

# constant for classes
classes = (0,1,2,3,4,5)

if graph_CM:
    # Build confusion matrix
    cf_matrix = confusion_matrix(test_label, test_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(conf_matrix_path)

if graph_F1:
    # Graphs the macro f1 metrics
    plt.plot(epoches, test_f1, label="Test f1 score")
    plt.plot(epoches, train_f1, label="Train  f1 score")
    plt.legend()
    plt.savefig(train_test_f1_path)
    plt.show()
    plt.close()