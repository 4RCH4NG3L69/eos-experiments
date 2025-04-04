import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.activation import get_activation
from utils.initialization import initialize_weights

# Basic fully-connected network with configurable width and depth
def create_fc_network(input_size, output_size, hidden_sizes, activation='relu', init_method='kaiming'):
    layers = []
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    for i in range(len(layer_sizes) - 1):
        layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
        initialize_weights(layer, init_method)
        layers.append(layer)
        
        # Don't add activation after the output layer
        if i < len(layer_sizes) - 2:
            layers.append(get_activation(activation))
    
    return nn.Sequential(*layers)


# Basic CNN with configurable channels, kernel sizes, and pooling
def create_cnn(input_channels, output_size, hidden_channels, kernel_size=3, 
               activation='relu', init_method='kaiming', pooling='max'):
    layers = []
    channels = [input_channels] + hidden_channels
    
    for i in range(len(channels) - 1):
        # Convolutional layer
        conv = nn.Conv2d(channels[i], channels[i+1], kernel_size, padding=kernel_size//2)
        initialize_weights(conv, init_method)
        layers.append(conv)
        layers.append(get_activation(activation))
        
        # Pooling layer
        if pooling.lower() == 'max':
            layers.append(nn.MaxPool2d(2, 2))
        elif pooling.lower() == 'avg':
            layers.append(nn.AvgPool2d(2, 2))
    
    # Flatten and add fully-connected output layer
    layers.append(nn.Flatten())
    
    # Calculate the size after convolutions and pooling
    # Assuming square input with size 28x28 (MNIST) or 32x32 (CIFAR)
    if input_channels == 1:  # MNIST
        size = 28
    else:  # CIFAR
        size = 32
    
    # Each pooling layer reduces size by half
    for _ in range(len(hidden_channels)):
        size //= 2
    
    fc_input_size = hidden_channels[-1] * size * size
    fc = nn.Linear(fc_input_size, output_size)
    initialize_weights(fc, init_method)
    layers.append(fc)
    
    return nn.Sequential(*layers)


# VGG implementation following the paper's approach
def create_vgg(input_channels, output_size, config='A', batch_norm=False, 
               activation='relu', init_method='kaiming'):
    # VGG configurations (A-E from the original paper)
    configs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    
    if config not in configs:
        raise ValueError(f"VGG configuration '{config}' not recognized")
    
    features = []
    in_channels = input_channels
    
    # Build feature extractor
    for v in configs[config]:
        if v == 'M':
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            initialize_weights(conv, init_method)
            features.append(conv)
            
            if batch_norm:
                features.append(nn.BatchNorm2d(v))
                
            features.append(get_activation(activation))
            in_channels = v
    
    # Create feature extractor
    feature_extractor = nn.Sequential(*features)
    
    # Calculate classifier input size
    if input_channels == 1:  # MNIST
        feature_size = 512 * (28 // (2 ** 5)) * (28 // (2 ** 5))  # 5 max-pooling layers
    else:  # CIFAR
        feature_size = 512 * (32 // (2 ** 5)) * (32 // (2 ** 5))  # 5 max-pooling layers
    
    # Create classifier
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_size, 4096),
        get_activation(activation),
        nn.Linear(4096, 4096),
        get_activation(activation),
        nn.Linear(4096, output_size)
    )
    
    # Initialize classifier weights
    for module in classifier:
        if isinstance(module, nn.Linear):
            initialize_weights(module, init_method)
    
    # Combine feature extractor and classifier
    return nn.Sequential(feature_extractor, classifier)


# Basic block for ResNet
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu', init_method='kaiming'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = get_activation(activation)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Initialize weights
        initialize_weights(self.conv1, init_method)
        initialize_weights(self.conv2, init_method)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            initialize_weights(self.shortcut[0], init_method)
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


# ResNet implementation following the paper's approach
def create_resnet(input_channels, output_size, block=BasicBlock, num_blocks=[2, 2, 2, 2], 
                  activation='relu', init_method='kaiming'):
    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.activation = get_activation(activation)
            
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            initialize_weights(self.conv1, init_method)
            
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            
            # Calculate final feature size
            if input_channels == 1:  # MNIST
                feature_size = 512 * block.expansion * (28 // 8) * (28 // 8)
            else:  # CIFAR
                feature_size = 512 * block.expansion * (32 // 8) * (32 // 8)
                
            self.linear = nn.Linear(feature_size, output_size)
            initialize_weights(self.linear, init_method)
        
        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride, activation, init_method))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.activation(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
    
    return ResNet()


# Factory function to create networks
def create_network(architecture, input_shape, output_size, activation='relu', 
                   init_method='kaiming', **kwargs):
    if len(input_shape) == 1:  # Flat input (FC network)
        input_size = input_shape[0]
        
        if architecture.lower() == 'fc':
            hidden_sizes = kwargs.get('hidden_sizes', [100, 100])
            return create_fc_network(input_size, output_size, hidden_sizes, 
                                    activation, init_method)
    else:  # Image input
        input_channels = input_shape[0]
        
        if architecture.lower() == 'fc':
            # Flatten image input for FC network
            input_size = input_shape[0] * input_shape[1] * input_shape[2]
            hidden_sizes = kwargs.get('hidden_sizes', [100, 100])
            
            # Add a flatten layer at the beginning
            layers = [nn.Flatten()]
            layers.extend(create_fc_network(input_size, output_size, hidden_sizes, 
                                           activation, init_method))
            return nn.Sequential(*layers)
        
        elif architecture.lower() == 'cnn':
            hidden_channels = kwargs.get('hidden_channels', [32, 64])
            kernel_size = kwargs.get('kernel_size', 3)
            pooling = kwargs.get('pooling', 'max')
            return create_cnn(input_channels, output_size, hidden_channels, kernel_size, 
                             activation, init_method, pooling)
        
        elif architecture.lower() == 'vgg':
            config = kwargs.get('config', 'A')
            batch_norm = kwargs.get('batch_norm', False)
            return create_vgg(input_channels, output_size, config, batch_norm, 
                             activation, init_method)
        
        elif architecture.lower() == 'resnet':
            num_blocks = kwargs.get('num_blocks', [2, 2, 2, 2])
            return create_resnet(input_channels, output_size, BasicBlock, num_blocks, 
                                activation, init_method)
    
    raise ValueError(f"Architecture '{architecture}' not recognized or incompatible with input shape")