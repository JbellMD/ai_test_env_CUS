import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def generate_scale_free_graph(num_nodes=100):
    G = nx.scale_free_graph(num_nodes)
    return from_networkx(G)