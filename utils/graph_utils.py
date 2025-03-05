import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any

class GraphUtils:
    @staticmethod
    def visualize_graph(graph: Dict[str, Any], title: str = "Graph Visualization") -> None:
        """Visualize a graph using NetworkX"""
        G = nx.Graph()
        
        # Add nodes
        for node, attrs in graph.get('nodes', {}).items():
            G.add_node(node, **attrs)
            
        # Add edges
        for edge, attrs in graph.get('edges', {}).items():
            G.add_edge(*edge, **attrs)
            
        # Draw graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue")
        plt.title(title)
        plt.show()
    
    @staticmethod
    def calculate_graph_metrics(graph: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various graph metrics"""
        G = nx.Graph()
        
        # Add nodes and edges
        for node, attrs in graph.get('nodes', {}).items():
            G.add_node(node, **attrs)
        for edge, attrs in graph.get('edges', {}).items():
            G.add_edge(*edge, **attrs)
            
        return {
            'density': nx.density(G),
            'clustering_coefficient': nx.average_clustering(G),
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G)
        }