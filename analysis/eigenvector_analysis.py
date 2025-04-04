import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector
import matplotlib.cm as cm


# Create a mapping from flattened parameter indices to named parameters
def create_parameter_mapping(model):
    mapping = {}
    flat_idx = 0
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        shape = param.shape
        
        for i in range(param_size):
            # Convert flat index to parameter-specific coordinates
            coord = np.unravel_index(i, shape)
            mapping[flat_idx + i] = {
                'name': name,
                'layer': name.split('.')[0] if '.' in name else name,
                'coordinates': coord,
                'shape': shape,
                'type': 'bias' if name.endswith('bias') else 'weight'
            }
        
        flat_idx += param_size
    
    return mapping


# Extract layer info from model
def extract_layer_info(model):
    layer_info = {}
    layer_sizes = {}
    
    for name, param in model.named_parameters():
        layer = name.split('.')[0] if '.' in name else name
        param_type = 'bias' if name.endswith('bias') else 'weight'
        
        if layer not in layer_info:
            layer_info[layer] = {'weights': 0, 'biases': 0, 'total': 0}
            
        if param_type == 'weight':
            layer_info[layer]['weights'] += param.numel()
        else:
            layer_info[layer]['biases'] += param.numel()
            
        layer_info[layer]['total'] += param.numel()
        layer_sizes[name] = param.numel()
    
    return layer_info, layer_sizes


# Analyze contributions of an eigenvector to model parameters
def analyze_eigenvector(eigenvector, param_mapping, top_k = 10, by_layer = False):
    # Get absolute contributions
    abs_contrib = np.abs(eigenvector.cpu().numpy())
    
    # If analyzing by layer, sum contributions per layer
    if by_layer:
        layer_contributions = {}
        
        for idx, val in enumerate(abs_contrib):
            layer = param_mapping[idx]['layer']
            if layer not in layer_contributions:
                layer_contributions[layer] = 0
            layer_contributions[layer] += val
            
        return layer_contributions
    
    # Otherwise, find individual top contributors
    top_indices = np.argsort(abs_contrib)[-top_k:][::-1]
    
    top_contributors = []
    for idx in top_indices:
        param_info = param_mapping[idx]
        contribution = float(eigenvector[idx])
        
        top_contributors.append({
            'index': idx,
            'param_name': param_info['name'],
            'layer': param_info['layer'],
            'coordinates': param_info['coordinates'],
            'contribution': contribution,
            'abs_contribution': abs(contribution),
            'param_type': param_info['type']
        })
        
    return top_contributors


# Create a visualization of eigenvector contributions
def visualize_eigenvector_contributions(model, eigenvectors, eigenvalues, 
                                       top_k = 5, max_eigenvectors = 6, figsize = (15, 12)):

    param_mapping = create_parameter_mapping(model)
    layer_info, layer_sizes = extract_layer_info(model)
    
    # Limit number of eigenvectors to analyze
    n_eigs = min(len(eigenvalues), max_eigenvectors)
    eigenvectors = eigenvectors[:, :n_eigs]
    eigenvalues = eigenvalues[:n_eigs]
    
    fig = plt.figure(figsize=figsize)
    
    # 1. Layer-wise contribution heatmap (most digestible representation)
    plt.subplot(2, 1, 1)
    
    # Compute layer contributions for each eigenvector
    layer_names = list(layer_info.keys())
    layer_contributions = np.zeros((len(layer_names), n_eigs))
    
    for i in range(n_eigs):
        layer_contribs = analyze_eigenvector(eigenvectors[:, i], param_mapping, by_layer=True)
        for j, layer in enumerate(layer_names):
            if layer in layer_contribs:
                # Normalize by layer size to account for different parameter counts
                layer_contributions[j, i] = layer_contribs[layer] / layer_info[layer]['total']
    
    # Create heatmap
    im = plt.imshow(layer_contributions, cmap='viridis')
    plt.colorbar(im, label='Normalized contribution')
    
    # Add labels
    plt.yticks(range(len(layer_names)), layer_names)
    plt.xticks(range(n_eigs), [f'λ{i+1}={eigenvalues[i]:.2f}' for i in range(n_eigs)])
    
    plt.title('Layer-wise Contributions to Principal Eigenvectors')
    plt.xlabel('Eigenvectors (with eigenvalues)')
    plt.ylabel('Network Layers')
    
    # 2. Top parameter contributions for first eigenvector
    plt.subplot(2, 1, 2)
    
    top_contributors = analyze_eigenvector(eigenvectors[:, 0], param_mapping, top_k=top_k)
    
    param_labels = [f"{c['layer']}\n{c['param_type']}{c['coordinates']}" for c in top_contributors]
    values = [c['abs_contribution'] for c in top_contributors]
    colors = ['g' if c['contribution'] > 0 else 'r' for c in top_contributors]
    
    # Create bar chart
    bars = plt.bar(range(len(param_labels)), values, color=colors)
    
    plt.xticks(range(len(param_labels)), param_labels, rotation=45, ha='right')
    plt.title(f'Top {top_k} Contributors to First Eigenvector (λ = {eigenvalues[0]:.4f})')
    plt.ylabel('Absolute Contribution')
    plt.xlabel('Parameter')
    
    # Add a legend for the colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='g', label='Positive contribution'),
        Patch(facecolor='r', label='Negative contribution')
    ]

    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Generate a comprehensive analysis of principal parameters
def analyze_principal_parameters(model, eigenvectors, eigenvalues, max_eigenvectors=6):

    # Create parameter mapping
    param_mapping = create_parameter_mapping(model)
    layer_info, _ = extract_layer_info(model)
    
    # Limit number of eigenvectors to analyze
    n_eigs = min(len(eigenvalues), max_eigenvectors)
    
    # Analyze each eigenvector
    results = {
        'eigenvalues': eigenvalues[:n_eigs].tolist(),
        'layer_structure': layer_info,
        'top_contributors': {},
        'layer_contributions': {}
    }
    
    for i in range(n_eigs):
        # Top individual contributors
        top_contributors = analyze_eigenvector(eigenvectors[:, i], param_mapping, top_k=10)
        results['top_contributors'][i] = top_contributors
        
        # Layer contributions
        layer_contribs = analyze_eigenvector(eigenvectors[:, i], param_mapping, by_layer=True)
        results['layer_contributions'][i] = layer_contribs
    
    # Also visualize the results
    visualize_eigenvector_contributions(model, eigenvectors, eigenvalues, 
                                        max_eigenvectors=max_eigenvectors)
    
    return results