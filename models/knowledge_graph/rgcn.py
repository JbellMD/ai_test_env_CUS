import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, hidden_dim)
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        
        x = self.embedding(x)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        
        return torch.sigmoid(x)