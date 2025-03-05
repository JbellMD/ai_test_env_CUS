import numpy as np
from typing import Dict, Any

class RewardSystem:
    def __init__(self):
        self.reward_weights = {
            'performance': 0.6,
            'stability': 0.3,
            'complexity': -0.1,
            'novelty': 0.2
        }
        self.history = []
        
    def calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calculate reward based on performance metrics"""
        reward = 0.0
        for key, weight in self.reward_weights.items():
            reward += metrics.get(key, 0.0) * weight
            
        self.history.append(reward)
        return reward
    
    def update_weights(self, recent_performance: float) -> None:
        """Adapt reward weights based on recent performance"""
        if recent_performance > np.mean(self.history[-10:]):
            # Increase focus on performance
            self.reward_weights['performance'] *= 1.1
        else:
            # Increase focus on stability
            self.reward_weights['stability'] *= 1.1
            
        # Normalize weights
        total = sum(self.reward_weights.values())
        for key in self.reward_weights:
            self.reward_weights[key] /= total