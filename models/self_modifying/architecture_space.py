from typing import List, Dict, Any
import random

class ArchitectureSpace:
    def __init__(self):
        self.operations = [
            'add_layer',
            'remove_layer',
            'modify_layer',
            'add_skip_connection',
            'change_activation'
        ]
        
    def generate_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidate modifications"""
        candidates = []
        for _ in range(10):
            operation = random.choice(self.operations)
            candidate = {
                'operation': operation,
                'config': self._generate_config(operation)
            }
            candidates.append(candidate)
        return candidates
        
    def _generate_config(self, operation: str) -> Dict[str, Any]:
        """Generate configuration for an operation"""
        if operation == 'add_layer':
            return {
                'type': random.choice(['linear', 'conv']),
                'size': random.randint(32, 512)
            }
        elif operation == 'modify_layer':
            return {
                'layer_index': random.randint(0, 5),
                'new_size': random.randint(32, 512)
            }
        else:
            return {}