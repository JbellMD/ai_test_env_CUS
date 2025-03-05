from torch_geometric.datasets import FB15k
import torch

def load_fb15k():
    dataset = FB15k(root='data/FB15k')
    return dataset