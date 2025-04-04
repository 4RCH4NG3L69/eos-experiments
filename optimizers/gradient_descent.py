import torch
from torch.optim import Optimizer


# Full-batch gradient descent optimizer
class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(GradientDescent, self).__init__(params, defaults)
    
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad.data
                
                # Simple weight update
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


# Stochastic gradient descent optimizer
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(SGD, self).__init__(params, defaults)
    
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss


# Create full-batch gradient descent function
def create_gradient_descent(params, lr=0.01):
    return GradientDescent(params, lr=lr)


# Create stochastic gradient descent function
def create_sgd(params, lr=0.01, batch_size=None, momentum=0, weight_decay=0, nesterov=False):
    return SGD(
        params, 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay, 
        nesterov=nesterov
    )


# Create Polyak momentum gradient descent
def create_polyak_momentum(params, lr=0.01, beta=0.9, weight_decay=0):
    return SGD(
        params,
        lr=lr,
        momentum=beta,
        dampening=0,
        weight_decay=weight_decay,
        nesterov=False
    )


# Create Nesterov momentum gradient descent
def create_nesterov_momentum(params, lr=0.01, beta=0.9, weight_decay=0):
    return SGD(
        params,
        lr=lr,
        momentum=beta,
        dampening=0,
        weight_decay=weight_decay,
        nesterov=True
    )