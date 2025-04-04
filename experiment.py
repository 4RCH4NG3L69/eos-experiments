import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import parameters_to_vector

# Import modules
from models import get_model
from optimizers import create_optimizer
from data.data_utils import load_dataset, get_data_loaders, get_device
from analysis.hessian import (
    get_hessian_eigenvalues, 
    get_hessian_eigenvalues_and_vectors, 
    compute_trajectory_length
)
from analysis.eigenvector_analysis import analyze_principal_parameters


# Main experiment function
def run_experiment(
    # Architecture
    architecture='fc',
    architecture_params=None,
    
    # Dataset
    dataset='cifar10',
    subset_size=5000,
    seed=42,
    
    # Model configuration
    activation='relu',
    loss_type='ce',
    init_method='kaiming',
    
    # Training parameters
    optimizer_type='gd',
    learning_rate=0.01,
    batch_size=None,  # None for full-batch
    max_iterations=1000,
    
    # Analysis parameters
    n_eigenvalues=1,
    
    **optimizer_params
):
    """
    
    Args:
        architecture (str): Model architecture ('fc', 'cnn', 'vgg', 'resnet', 'transformer')
        architecture_params (dict): Architecture-specific parameters
        dataset (str): Dataset name ('mnist', 'cifar10')
        subset_size (int): Size of dataset subset (None for full dataset)
        seed (int): Random seed for reproducibility
        activation (str): Activation function ('relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu')
        loss_type (str): Loss function ('ce' for cross-entropy, 'mse' for mean squared error)
        init_method (str): Weight initialization method ('uniform', 'xavier', 'kaiming', 'zeros', 'ones', 'normal')
        optimizer_type (str): Optimizer type ('gd', 'sgd', 'adam', 'mirror', 'polyak', 'nesterov')
        learning_rate (float): Learning rate
        batch_size (int): Batch size (None for full-batch)
        max_iterations (int): Maximum number of iterations
        n_eigenvalues (int): Number of top eigenvalues to compute
        **optimizer_params: Additional optimizer-specific parameters
    
    Returns:
        dict: Dictionary containing results and model
    """
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set default architecture parameters if None
    if architecture_params is None:
        architecture_params = {}
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset, loss_type, subset_size, seed)
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size)
    
    # Determine input and output dimensions
    if dataset.lower() == 'mnist':
        input_shape = (1, 28, 28)
        output_size = 10
    else:  # CIFAR-10
        input_shape = (3, 32, 32)
        output_size = 10
    
    # Create model
    model = get_model(
        architecture=architecture,
        input_shape=input_shape,
        output_size=output_size,
        activation=activation,
        init_method=init_method,
        **architecture_params
    ).to(device)
    
    # Create optimizer
    optimizer = create_optimizer(
        optimizer_type=optimizer_type,
        params=model.parameters(),
        learning_rate=learning_rate,
        batch_size=batch_size,
        **optimizer_params
    )
    
    # Create loss function
    if loss_type.lower() == 'ce':
        criterion = nn.CrossEntropyLoss()
    else:  # MSE
        criterion = nn.MSELoss()
    
    # Initialize tracking variables
    train_loss_history = []
    train_acc_history = []
    eigenvalues_history = []
    parameter_history = []
    
    # Store initial parameters
    parameter_history.append(parameters_to_vector(model.parameters()).detach().clone())
    
    # Compute initial eigenvalues
    initial_eigenvalues = get_hessian_eigenvalues(
        model, criterion, train_dataset, 
        neigs=n_eigenvalues, 
        physical_batch_size=1000
    )
    eigenvalues_history.append(initial_eigenvalues)
    
    # Training loop
    print("Starting training...")
    for iteration in range(max_iterations):
        # Zero gradients
        optimizer.zero_grad()
        
        # Full batch gradient for computing loss/gradient
        total_loss = 0
        correct = 0
        total = 0
        
        # Process batches
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if loss_type.lower() == 'ce':
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
            else:  # MSE
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                _, target_cls = torch.max(targets.data, 1)
                correct += (predicted == target_cls).sum().item()
                total += targets.size(0)
            
            if len(train_loader) > 1:  # If using mini-batches
                loss.backward()
                total_loss += loss.item() * inputs.size(0)
            else:  # Full batch
                total_loss = loss.item() * inputs.size(0)
                loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Store current parameters for trajectory calculation
        parameter_history.append(parameters_to_vector(model.parameters()).detach().clone())
        
        # Compute and store metrics
        avg_loss = total_loss / len(train_dataset)
        accuracy = 100.0 * correct / total
        
        train_loss_history.append(avg_loss)
        train_acc_history.append(accuracy)
        
        # Compute eigenvalues at every iteration
        evals = get_hessian_eigenvalues(
            model, criterion, train_dataset, 
            neigs=n_eigenvalues, 
            physical_batch_size=1000
        )
        eigenvalues_history.append(evals)
        
        # Print iteration progress, loss and top eigenvalue
        print(f"Iteration {iteration+1}/{max_iterations}, loss: {avg_loss:.6f}, top eigenvalue: {evals[0].item():.6f}")
    
    print("Training complete. Generating visualizations...")
    
    # Compute trajectory length
    traj_length = compute_trajectory_length(parameter_history)
    
    # Make sure traj_length matches the number of iterations
    if len(traj_length) > max_iterations:
        traj_length = traj_length[:max_iterations]
    elif len(traj_length) < max_iterations:
        # Should not happen, but just in case
        traj_length = traj_length + [traj_length[-1]] * (max_iterations - len(traj_length))
    
    # Compute final eigenvalues and eigenvectors for analysis
    final_eigenvalues, final_eigenvectors = get_hessian_eigenvalues_and_vectors(
        model, criterion, train_dataset, 
        neigs=n_eigenvalues,
        physical_batch_size=1000
    )
    
    # Generate visualizations
    _visualize_training_results(
        train_loss_history,
        train_acc_history,
        eigenvalues_history,
        traj_length,
        learning_rate
    )
    
    # Perform eigenvector analysis
    print("Analyzing principal parameters...")
    analysis_results = analyze_principal_parameters(
        model, final_eigenvectors, final_eigenvalues, 
        max_eigenvectors=min(n_eigenvalues, 6)
    )
    
    # Return results
    return {
        'model': model,
        'metrics': {
            'train_loss': train_loss_history,
            'train_accuracy': train_acc_history,
            'eigenvalues': eigenvalues_history,
            'trajectory_length': traj_length
        },
        'analysis': analysis_results,
        'final_eigenvalues': final_eigenvalues,
        'final_eigenvectors': final_eigenvectors
    }


# Visualize training results
def _visualize_training_results(
    train_loss, train_accuracy, eigenvalues, trajectory_length, 
    learning_rate
):
    iterations = list(range(len(train_loss)))
    eig_iterations = list(range(len(eigenvalues)))
    
    # 1. Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss)
    plt.title('Training Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 2. Plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_accuracy)
    plt.title('Training Accuracy vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show()
    
    # 3. Plot eigenvalues
    if len(eigenvalues) > 0:
        plt.figure(figsize=(10, 6))
        eigenvalues_array = torch.stack(eigenvalues).numpy()
        
        for i in range(eigenvalues_array.shape[1]):
            plt.plot(eig_iterations, eigenvalues_array[:, i], label=f'λ{i+1}')
        
        plt.axhline(y=2/learning_rate, linestyle='--', color='black', label='2/η')
        plt.title('Top Eigenvalues of Hessian vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Eigenvalue')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 4. Plot trajectory length
    plt.figure(figsize=(10, 6))
    # Ensure they have the same length for plotting
    plot_length = min(len(iterations), len(trajectory_length))
    plt.plot(iterations[:plot_length], trajectory_length[:plot_length])
    plt.title('Optimization Trajectory Length vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Parameter Distance')
    plt.grid(True)
    plt.show()


# Helper function to create simplified experiment interface
def edge_of_stability_experiment(architecture, dataset, optimizer, learning_rate, max_iterations=1000, **kwargs):
    """
    Simplified interface for running Edge of Stability experiments.
    
    Args:
        architecture (str): Model architecture ('fc', 'cnn', 'vgg', 'resnet', 'transformer')
        dataset (str): Dataset name ('mnist', 'cifar10')
        optimizer (str): Optimizer type ('gd', 'sgd', 'adam', 'mirror')
        learning_rate (float): Learning rate to use
        max_iterations (int): Maximum number of iterations
        **kwargs: Additional parameters
        
    Returns:
        dict: Results dictionary
    """
    return run_experiment(
        architecture=architecture,
        dataset=dataset,
        optimizer_type=optimizer,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **kwargs
    )