import torch
import torch.nn as nn

import utils

class BaseViT(nn.Module):
    """Base class for Vision Transformers with configurable activation functions."""

    def __init__(self, input_length, patch_size, in_channels, num_layers, d_model, num_heads, d_ff, dropout, activation):
        """
        Args:
            input_length (int): Input sequence length.
            patch_size (int): Size of each image patch.
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_layers (int): Number of transformer encoder layers.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feedforward network dimension.
            dropout (float): Dropout probability.
            activation (nn.Module): activation function for transformer layers.
        """
        super(BaseViT, self).__init__()

        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.d_model = d_model

        self.patch_embedding = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=self.num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(nn.Sequential(encoder_layer), num_layers=num_layers)

        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)  # Single output (regression) or change num_classes for classification
        )

        self.config = utils.read_config()
        if self.config.get("init"):
            self.initialize_weights()

    def initialize_weights(self):
        """Applies weight initialization based on activation functions."""
        for layer, act in zip(self.layers, self.activations):
            if isinstance(act, (nn.ReLU, nn.LeakyReLU)):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(act, (nn.Sigmoid, nn.Tanh)):
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight)
        nn.init.xavier_uniform_(self.patch_embedding.weight)

    def forward(self, x):
        batch_size = x.shape[0]

        # Convert image into patch embeddings
        x = self.patch_embedding(x)  # [B, d_model, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, d_model]

        # Append CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # Shape: [B, num_patches + 1, d_model]

        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x[:, 0]  # Extract [CLS] token
        return self.mlp_head(x)

class PositionalEncoding(nn.Module):
    """Applies sinusoidal positional encoding to input sequences."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NonOverlappingViT(BaseViT):
    """Vision Transformer with standard non-overlapping patch processing."""
    
    def __init__(self, act):
        super().__init__(
            input_length=12,
            patch_size=3,
            in_channels=1,
            num_layers=3,
            d_model=768,
            num_heads=4,
            d_ff=512,
            dropout=0.1,
            activations=act
        )