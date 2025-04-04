import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.activation import get_activation
from utils.initialization import initialize_weights


# Position encoder for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer to store the positional encoding
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# Patch embedding for vision transformer
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, init_method='kaiming'):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection layer
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        initialize_weights(self.proj, init_method)
        
    def forward(self, x):
        # Expected input shape: (batch_size, in_channels, img_size, img_size)
        # Output shape: (batch_size, n_patches, embed_dim)
        
        # Apply patch projection
        x = self.proj(x)  # (B, embed_dim, img_size//patch_size, img_size//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        
        return x


# Transformer for image classification
class VisionTransformer(nn.Module):
    def __init__(self, 
                img_size=32, 
                patch_size=4, 
                in_channels=3, 
                num_classes=10,
                embed_dim=256, 
                depth=6, 
                num_heads=8, 
                mlp_ratio=4.0,
                dropout=0.1,
                attn_dropout=0.0,
                activation='gelu',
                init_method='kaiming'):
        super(VisionTransformer, self).__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            init_method=init_method
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        initialize_weights(self.fc, init_method)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        # Get patches
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Process with transformer
        x = self.transformer(x)
        
        # Use [CLS] token for classification
        x = self.norm(x[:, 0])
        x = self.fc(x)
        
        return x


# Factory function to create a transformer
def create_transformer(input_shape, output_size, activation='gelu', init_method='kaiming', **kwargs):
    in_channels = input_shape[0]
    img_size = input_shape[1]  # Assuming square image
    
    # Default configuration based on input size
    if img_size == 28:  # MNIST
        patch_size = 4
        embed_dim = 192
        depth = 4
        num_heads = 4
    else:  # CIFAR-10 or similar
        patch_size = 4
        embed_dim = 256
        depth = 6
        num_heads = 8
    
    # Override with kwargs if provided
    patch_size = kwargs.get('patch_size', patch_size)
    embed_dim = kwargs.get('embed_dim', embed_dim)
    depth = kwargs.get('depth', depth)
    num_heads = kwargs.get('num_heads', num_heads)
    mlp_ratio = kwargs.get('mlp_ratio', 4.0)
    dropout = kwargs.get('dropout', 0.1)
    attn_dropout = kwargs.get('attn_dropout', 0.0)
    
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=output_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        activation=activation,
        init_method=init_method
    )