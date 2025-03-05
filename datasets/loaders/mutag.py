from torch_geometric.datasets import TUDataset
import torch

def load_mutag():
    dataset = TUDataset(root='data/MUTAG', name='MUTAG')
    return dataset