import torch
from torch.optim import Optimizer
import math


# Mirror Descent optimizer
class MirrorDescent(Optimizer):
    def __init__(self, params, lr=0.01, dgf='squared_l2', epsilon=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Validate distance generating function
        valid_dgfs = ['squared_l2', 'neg_entropy']
        if dgf not in valid_dgfs:
            raise ValueError(f"Invalid dgf: {dgf}. Must be one of {valid_dgfs}")
        
        defaults = dict(lr=lr, dgf=dgf, epsilon=epsilon)
        super(MirrorDescent, self).__init__(params, defaults)
        
        # Initialize dual variables for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    if dgf == 'squared_l2':
                        # For squared L2, the dual variable is just the parameter itself
                        state['dual'] = p.data.clone()
                    elif dgf == 'neg_entropy':
                        # For negative entropy, the dual variable is log(p)
                        # Ensure parameter values are positive before taking log
                        state['dual'] = torch.log(torch.clamp(p.data, min=epsilon))
    
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            dgf = group['dgf']
            epsilon = group['epsilon']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Mirror descent step based on the chosen distance generating function
                if dgf == 'squared_l2':
                    # Standard gradient descent update 
                    # 1. Map to dual space (identity for L2)
                    # 2. Update in dual space: θ_dual_{t+1} = θ_dual_t - η∇f(θ_t)
                    # 3. Map back to primal space (identity for L2)
                    p.data.add_(grad, alpha=-lr)
                    state['dual'] = p.data.clone()
                    
                elif dgf == 'neg_entropy':
                    # 1. Already in dual space (log space)
                    # 2. Update in the dual space: θ_dual_{t+1} = θ_dual_t - η∇f(θ_t)
                    state['dual'].add_(grad, alpha=-lr)
                    
                    # 3. Map back to the primal space: θ_{t+1} = exp(θ_dual_{t+1})
                    p.data = torch.exp(state['dual'])
        
        return loss


def create_mirror_descent(params, lr=0.01, dgf='squared_l2'):
    return MirrorDescent(params, lr=lr, dgf=dgf)