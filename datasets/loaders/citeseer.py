from torch_geometric.datasets import Planetoid
import torch

def load_citeseer():
    dataset = Planetoid(root='data/Citeseer', name='Citeseer')
    data = dataset[0]
    return {
        'x': data.x,
        'edge_index': data.edge_index,
        'y': data.y,
        'train_mask': data.train_mask,
        'val_mask': data.val_mask,
        'test_mask': data.test_mask
    }