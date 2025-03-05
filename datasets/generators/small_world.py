import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def generate_small_world_graph(num_nodes=100, k=4, p=0.1):
    G = nx.watts_strogatz_graph(num_nodes, k, p)
    return from_networkx(G)