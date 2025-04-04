import torch
import torch.nn as nn
import math


# Initialize weights of a layer based on specified method
def initialize_weights(layer, method='kaiming'):
    if not hasattr(layer, 'weight'):
        return
    
    method = method.lower()
    
    # Initialize weights based on method
    if method == 'uniform':
        nn.init.uniform_(layer.weight, -0.1, 0.1)
    elif method == 'xavier':
        nn.init.xavier_uniform_(layer.weight)
    elif method == 'kaiming':
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    elif method == 'zeros':
        nn.init.zeros_(layer.weight)
    elif method == 'ones':
        nn.init.ones_(layer.weight)
    elif method == 'normal':
        nn.init.normal_(layer.weight, mean=0, std=0.01)
    else:
        raise ValueError(f"Initialization method '{method}' not recognized")
    
    # Initialize bias if it exists
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)


# Check if initialization method is supported
def initialization_exists(method):
    return method.lower() in {
        'uniform', 'xavier', 'kaiming', 'zeros', 'ones', 'normal'
    }


# Apply initialization to all layers in a model
def initialize_model(model, method='kaiming'):
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            initialize_weights(module, method)