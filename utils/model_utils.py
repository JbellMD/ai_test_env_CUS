import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np

class ModelUtils:
    @staticmethod
    def calculate_model_size(model: nn.Module) -> int:
        """Calculate the size of a model in bytes"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    @staticmethod
    def calculate_model_complexity(model: nn.Module) -> Dict[str, float]:
        """Calculate various complexity metrics for a model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_bytes': ModelUtils.calculate_model_size(model),
            'depth': ModelUtils.calculate_model_depth(model)
        }
    
    @staticmethod
    def calculate_model_depth(model: nn.Module) -> int:
        """Calculate the depth of a model"""
        max_depth = 0
        for module in model.modules():
            if isinstance(module, nn.Module) and module is not model:
                max_depth = max(max_depth, len(list(module.children())))
        return max_depth
    
    @staticmethod
    def calculate_gradient_norms(model: nn.Module) -> Dict[str, float]:
        """Calculate gradient norms for each layer"""
        norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = torch.norm(param.grad).item()
        return norms
    
    @staticmethod
    def calculate_parameter_distribution(model: nn.Module) -> Dict[str, Any]:
        """Calculate parameter value distribution"""
        params = [p.detach().cpu().numpy() for p in model.parameters()]
        flat_params = np.concatenate([p.flatten() for p in params])
        
        return {
            'mean': float(np.mean(flat_params)),
            'std': float(np.std(flat_params)),
            'min': float(np.min(flat_params)),
            'max': float(np.max(flat_params)),
            'percentiles': {
                '25': float(np.percentile(flat_params, 25)),
                '50': float(np.percentile(flat_params, 50)),
                '75': float(np.percentile(flat_params, 75))
            }
        }
    
    @staticmethod
    def calculate_activation_stats(activations: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate statistics for layer activations"""
        flat_acts = torch.cat([a.flatten() for a in activations])
        return {
            'mean': float(flat_acts.mean()),
            'std': float(flat_acts.std()),
            'min': float(flat_acts.min()),
            'max': float(flat_acts.max())
        }