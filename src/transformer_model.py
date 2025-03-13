import torch
import torch.nn as nn

import utils

class TransformerModel(nn.Module):
    """Transformer model with configurable parameters."""
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, activation):
        """
        Args:
            num_layers (int): Number of transformer encoder layers.
            d_model (int): Dimensionality of the input embeddings.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of the feedforward network.
            dropout (float): Dropout probability.
            activation (nn.Module): Activation function to be used.
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Linear(d_model, d_model)  # Embedding layer
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.config = utils.read_config()
        
        if self.config.get("init"):
            self.initialize_weights()

    def initialize_weights(self):
        """Applies Xavier initialization to the embeddings."""
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.positional_encoding(x)
        return self.transformer_encoder(x)

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