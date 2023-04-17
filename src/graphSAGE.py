'''
Changes introduced:
Stratified train test split with 20% test examples

TODO:
Hidden layer dimensions, no of graphSAGE layers, #epochs are hyperparameters
Try different pooling functions
'''

import os
import pickle
import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, SAGEConv

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

print(dir)

networkx_graphs_path = dir + "/data/torch_graphs_041723_v2.pkl"
conf_matrix_path = dir + '/data/CM_graphS_041723.png'

if os.path.isfile(networkx_graphs_path):
    with open(networkx_graphs_path, "rb") as handle:
        print("Retrieve the pickle!")
        dataset = pickle.load(handle)

NUM_CLASSES = 6
NUM_NODE_FEATURES = 33

#Class label for each graph in labels
labels = np.empty(len(dataset))
for i in range(len(dataset)):
  labels[i] = dataset[i].y.item()

#labelCount stores the no of graphs for each class
labelCount = np.zeros(NUM_CLASSES)
#labelCount = {}
for i in range(len(labels)):
  #print(labels[i])
  # if(labels[i] in labelCount):
  #   labelCount[labels[i]]+=1
  # else:
  #   labelCount[labels[i]]=1
  labelCount[int(labels[i])]+=1

print(labelCount)

maxClassGraphs = np.amax(labelCount)

# for cl in labelCount:
#   print(cl,": ", labelCount[cl])

#Calculating class weights
classWeights = maxClassGraphs/labelCount
classWeights = torch.from_numpy(classWeights)
print(classWeights)

torch.manual_seed(12345)


train_dataset , test_dataset = train_test_split(dataset, test_size=0.2, stratify = labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train_dataset = train_dataset.to(device)
#train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lambda x: default_collate(x).to(device))
#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: default_collate(x).to(device))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# for step, data in enumerate(train_loader):
#   data = data.to(device)
#     print('For training data:')
#     print('=======')
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print('=======')
#     print(data)
#     print()


# for step, data in enumerate(test_loader):
#   data = data.to(device)
#     print('For test data:')
#     print('=======')
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print('=======')
#     print(data)
#     print()

# a simple graphSAGE model
class graphSAGE(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(NUM_NODE_FEATURES, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
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


model = graphSAGE(hidden=64)
model = model.to(device)
model = model.to(torch.float64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
classWeights = classWeights.to(device)
criterion = torch.nn.CrossEntropyLoss(weight = classWeights)

def train():
    model.train()

    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #return loss


def test(loader):
    model.eval()

    correct = 0
    y_pred_list = []
    y_label_list = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        y_pred_list += list(pred.data.cpu().numpy())
        y_label_list += list(data.y.data.cpu().numpy())
    #accuracy = correct / len(loader.dataset)
    f1Score = f1_score(y_label_list, y_pred_list, average = 'weighted')
    return f1Score, y_pred_list, y_label_list

final_test_pred = []
final_test_label = []
for epoch in range(1, 100):
    train()
    trainFScore, train_pred, train_label = test(train_loader)
    testFScore, test_pred, test_label = test(test_loader)

    print(f'Epoch: {epoch:03d}', 'Train F1 Score:', trainFScore, 'Test F1 Score: ', testFScore)
    if(epoch == 99):
      final_test_pred = test_pred
      final_test_label = test_label 

print("The final weighted F1 score is: ", f1_score(final_test_label, final_test_pred, average = 'weighted'))


# constant for classes
classes = (0,1,2,3,4,5)

# Build confusion matrix
cf_matrix = confusion_matrix(test_label, test_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sns.heatmap(df_cm, annot=True)
plt.savefig(conf_matrix_path)
'''
'''
