from torch import nn
from typing import List

# need extra parameter to specify a (wider!) sampling distribution. PyTorch's default initializaton samples from a 
# distribution that becomes smaller with growing parameters.


class FCNN(nn.Module):

    def __init__(self, activation: str, hidden_sizes: List[int]):

        super().__init__()
        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        layers = [nn.Flatten()]
        in_size = 28 * 28
        
        for hs in hidden_sizes:
            layers.append(nn.Linear(in_size, hs))
            layers.append(act)
            in_size = hs
            
        layers.append(nn.Linear(in_size, 10))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)