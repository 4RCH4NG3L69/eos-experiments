import torch
import numpy as np
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import LinearOperator, eigsh
from torch.utils.data import DataLoader

# Compute Hessian-vector product
def compute_hvp(model, loss_fn, dataset, vector, physical_batch_size=None, P=None):
    device = next(model.parameters()).device
    
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float)
    vector = vector.to(device)
    
    # Apply preconditioner if provided
    if P is not None:
        vector = vector / P.to(device).sqrt()
    
    if physical_batch_size is None or physical_batch_size >= len(dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data, targets = next(iter(loader))
        data, targets = data.to(device), targets.to(device)
        
        outputs = model(data)
        batch_size = data.size(0)
        loss = loss_fn(outputs, targets) * batch_size / len(dataset)
        
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        flat_grad = parameters_to_vector(grads)
        
        dot_product = torch.sum(flat_grad * vector)
        
        hvp = torch.autograd.grad(dot_product, model.parameters(), retain_graph=True)
        flat_hvp = parameters_to_vector(hvp)
        
    else:
        num_params = sum(p.numel() for p in model.parameters())
        flat_hvp = torch.zeros(num_params, device=device)
        
        loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=False)
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            batch_size = data.size(0)
            loss = loss_fn(outputs, targets) * batch_size / len(dataset)
            
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            flat_grad = parameters_to_vector(grads)
            
            dot_product = torch.sum(flat_grad * vector)
            
            hvp_batch = torch.autograd.grad(dot_product, model.parameters(), retain_graph=True)
            flat_hvp += parameters_to_vector(hvp_batch)
    
    # Apply preconditioner if provided
    if P is not None:
        flat_hvp = flat_hvp / P.to(device).sqrt()
        
    return flat_hvp

# Lanczos algorithm for eigenvalue computation
def lanczos(hvp_function, dim, neigs=1):
    def mv(v):
        v_tensor = torch.FloatTensor(v)
        if torch.cuda.is_available():
            v_tensor = v_tensor.cuda()
        elif torch.backends.mps.is_available():
            v_tensor = v_tensor.to("mps")
        else:
            v_tensor = v_tensor.cpu()
        
        # Get HVP and convert back to numpy
        hvp = hvp_function(v_tensor)
        return hvp.cpu().detach().numpy()
    
    operator = LinearOperator((dim, dim), matvec=mv)
    
    eigenvalues, eigenvectors = eigsh(operator, k=neigs, which='LM', tol=1e-8)
    
    # Sort by descending eigenvalue (like the original code)
    return torch.from_numpy(np.ascontiguousarray(eigenvalues[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(eigenvectors, -1)).copy()).float()

# Get top eigenvalues of the Hessian
def get_hessian_eigenvalues(model, loss_fn, dataset, neigs=1, physical_batch_size=None, P=None):
    num_params = sum(p.numel() for p in model.parameters())
    
    # Create HVP function closure
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, dataset, delta, physical_batch_size, P)
    
    # Compute eigenvalues
    eigenvalues, _ = lanczos(hvp_delta, num_params, neigs=neigs)
    return eigenvalues

# Get top eigenvalues and eigenvectors
def get_hessian_eigenvalues_and_vectors(model, loss_fn, dataset, neigs=1, physical_batch_size=None, P=None):
    num_params = sum(p.numel() for p in model.parameters())
    
    # Create HVP function closure
    hvp_delta = lambda delta: compute_hvp(model, loss_fn, dataset, delta, physical_batch_size, P)
    
    # Compute eigenvalues and eigenvectors
    return lanczos(hvp_delta, num_params, neigs=neigs)

# Compute trajectory length
def compute_trajectory_length(param_history):
    if len(param_history) <= 1:
        return [0.0] * len(param_history)
    
    distances = []
    for i in range(1, len(param_history)):
        dist = torch.norm(param_history[i] - param_history[i-1]).item()
        distances.append(dist)
    
    cum_distances = [0.0]
    cum_distances.extend(np.cumsum(distances).tolist())
    
    # Make sure the length is correct
    if len(cum_distances) > len(param_history):
        cum_distances = cum_distances[:len(param_history)]
    elif len(cum_distances) < len(param_history):
        cum_distances.extend([cum_distances[-1]] * (len(param_history) - len(cum_distances)))
    
    return cum_distances