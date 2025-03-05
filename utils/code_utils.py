import ast
import astor
from typing import Dict, Any

class CodeUtils:
    @staticmethod
    def parse_code(code: str) -> ast.AST:
        """Parse code into AST"""
        return ast.parse(code)
    
    @staticmethod
    def generate_code(tree: ast.AST) -> str:
        """Generate code from AST"""
        return astor.to_source(tree)
    
    @staticmethod
    def find_nodes(tree: ast.AST, node_type: type) -> list:
        """Find nodes of a specific type in the AST"""
        return [node for node in ast.walk(tree) if isinstance(node, node_type)]
    
    @staticmethod
    def modify_node(node: ast.AST, modifications: Dict[str, Any]) -> ast.AST:
        """Modify an AST node"""
        for key, value in modifications.items():
            setattr(node, key, value)
        return node
    
    @staticmethod
    def add_node(parent: ast.AST, new_node: ast.AST) -> ast.AST:
        """Add a new node to the AST"""
        if isinstance(parent, ast.Module):
            parent.body.append(new_node)
        elif hasattr(parent, 'body'):
            parent.body.append(new_node)
        return parent
    
    @staticmethod
    def remove_node(parent: ast.AST, node: ast.AST) -> ast.AST:
        """Remove a node from the AST"""
        if isinstance(parent, ast.Module):
            parent.body = [n for n in parent.body if n != node]
        elif hasattr(parent, 'body'):
            parent.body = [n for n in parent.body if n != node]
        return parent