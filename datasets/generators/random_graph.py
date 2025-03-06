import networkx as nx
import torch
from torch_geometric.utils import from_networkx

def generate_random_graph(num_nodes=100, edge_prob=0.1):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    return from_networkx(G)