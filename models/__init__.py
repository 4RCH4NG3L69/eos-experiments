from .architectures import create_network, create_fc_network, create_cnn, create_vgg, create_resnet
from .transformer import create_transformer

# Factory function to create models based on architecture type
def get_model(architecture, input_shape, output_size, activation='relu', init_method='kaiming', **kwargs):
    """
    Factory function to create a model with the specified architecture.
    
    Args:
        architecture (str): Type of architecture ('fc', 'cnn', 'vgg', 'resnet', 'transformer')
        input_shape (tuple): Shape of input data 
                             - For images: (channels, height, width)
                             - For flat data: (input_size,)
        output_size (int): Number of output classes/values
        activation (str): Activation function to use
        init_method (str): Weight initialization method
        **kwargs: Additional architecture-specific parameters
    
    Returns:
        nn.Module: The constructed model
    """
    architecture = architecture.lower()
    
    if architecture == 'transformer':
        return create_transformer(input_shape, output_size, activation, init_method, **kwargs)
    else:
        return create_network(architecture, input_shape, output_size, activation, init_method, **kwargs)