import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class Visualization:
    @staticmethod
    def plot_training_history(history: List[Dict[str, float]]) -> None:
        """Plot training history metrics"""
        epochs = range(1, len(history) + 1)
        metrics = history[0].keys()
        
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            values = [h[metric] for h in history]
            plt.plot(epochs, values, label=metric)
            
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_architecture_complexity(complexity_history: List[float]) -> None:
        """Plot architecture complexity over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(complexity_history, color='purple')
        plt.xlabel('Modification Step')
        plt.ylabel('Complexity')
        plt.title('Architecture Complexity Over Time')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def plot_performance_metrics(metrics: Dict[str, Any]) -> None:
        """Plot performance metrics"""
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='green')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        plt.grid(True)
        plt.show()