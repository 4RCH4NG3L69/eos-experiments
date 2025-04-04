import torch
from torch.optim import Adam

# Import optimizer implementations
from .gradient_descent import create_gradient_descent, create_sgd, create_polyak_momentum, create_nesterov_momentum
from .mirror_descent import create_mirror_descent


# Create an optimizer based on type and parameters
def create_optimizer(optimizer_type, params, **kwargs):
    """
    Args:
        optimizer_type (str): Type of optimizer ('gd', 'sgd', 'adam', 'mirror')
        params: Model parameters to optimize
        **kwargs: Optimizer-specific parameters
        
    """
    optimizer_type = optimizer_type.lower()
    
    # Extract common parameters
    lr = kwargs.get('learning_rate', 0.01)
    
    if optimizer_type == 'gd':
        return create_gradient_descent(params, lr=lr)
    
    elif optimizer_type == 'sgd':
        batch_size = kwargs.get('batch_size', None)
        momentum = kwargs.get('momentum', 0.0)
        weight_decay = kwargs.get('weight_decay', 0.0)
        nesterov = kwargs.get('nesterov', False)
        return create_sgd(params, lr=lr, batch_size=batch_size, 
                         momentum=momentum, weight_decay=weight_decay, 
                         nesterov=nesterov)
    
    elif optimizer_type == 'adam':
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        weight_decay = kwargs.get('weight_decay', 0.0)
        return Adam(params, lr=lr, betas=(beta1, beta2), 
                   eps=epsilon, weight_decay=weight_decay)
    
    elif optimizer_type == 'mirror':
        dgf = kwargs.get('dgf', 'squared_l2')
        return create_mirror_descent(params, lr=lr, dgf=dgf)
    
    elif optimizer_type == 'polyak':
        beta = kwargs.get('beta', 0.9)
        weight_decay = kwargs.get('weight_decay', 0.0)
        return create_polyak_momentum(params, lr=lr, beta=beta, 
                                     weight_decay=weight_decay)
    
    elif optimizer_type == 'nesterov':
        beta = kwargs.get('beta', 0.9)
        weight_decay = kwargs.get('weight_decay', 0.0)
        return create_nesterov_momentum(params, lr=lr, beta=beta, 
                                       weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' not recognized")


# Check if an optimizer type is supported
def optimizer_exists(optimizer_type):

    return optimizer_type.lower() in {
        'gd', 'sgd', 'adam', 'mirror', 'polyak', 'nesterov'
    }