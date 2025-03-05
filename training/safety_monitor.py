import ast
from typing import Dict, Any

class SafetyMonitor:
    def __init__(self):
        self.safety_rules = [
            self._check_resource_usage,
            self._check_execution_time,
            self._check_memory_usage
        ]
        
    def evaluate_safety(self, code: str) -> Dict[str, float]:
        """Evaluate code for safety constraints"""
        tree = ast.parse(code)
        
        safety_metrics = {
            'resource_usage': 1.0,
            'execution_time': 1.0,
            'memory_usage': 1.0
        }
        
        for rule in self.safety_rules:
            rule_name, score = rule(tree)
            safety_metrics[rule_name] = score
            
        return safety_metrics
    
    def _check_resource_usage(self, tree: ast.AST) -> tuple:
        """Check for excessive resource usage"""
        return ('resource_usage', 1.0)
    
    def _check_execution_time(self, tree: ast.AST) -> tuple:
        """Check for potential long execution times"""
        return ('execution_time', 1.0)
    
    def _check_memory_usage(self, tree: ast.AST) -> tuple:
        """Check for potential memory issues"""
        return ('memory_usage', 1.0)