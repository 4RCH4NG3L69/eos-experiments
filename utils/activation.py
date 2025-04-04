import torch
import torch.nn as nn
import torch.nn.functional as F


# Get a PyTorch activation function by name
def get_activation(activation_name):
    activation_name = activation_name.lower()
    
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=1),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01)
    }
    
    if activation_name not in activations:
        raise ValueError(f"Activation function '{activation_name}' not recognized")
    
    return activations[activation_name]


# Check if an activation function is supported
def activation_exists(activation_name):
    return activation_name.lower() in {
        'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu'
    }