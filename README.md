# CS594 project
## Packages: 
- ```pip intsall pandas```
- ```pip install openpyxl```
- ```pip install biopython```
- ```pip install networkx ```
- ```pip install torch torchvision```
- ```pip install torch_geometric```

## Structure:
```src``` folder contains:
- ```gcn.py```: Contains the neural network model
- ```main.py```: Contains scripts to generate dataset
- ```utils.py```: Contains method to use to generate dataset


```data``` folder contains:
- ```graphs_040423.pkl```: unpickle this will return ```graphs, labels```. Graphs: A Python list contains graphs in NetWorkX format, and labels: contains the label of the graphs list. 

## TODO 
- [ ] Loss function
- [ ] MLP before/ aftrer GNN 
- [ ] Evaluation metrics
- [ ] different model: GAT, GIN GraphSAGE 
- [ ] balancing the data if possible 
