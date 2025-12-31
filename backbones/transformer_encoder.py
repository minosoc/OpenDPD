__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention is All You Need'"""
    
    def __init__(self, d_model, max_len=10000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create initial positional encoding matrix
        pe = self._create_pe(max_len, d_model)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def _create_pe(self, length, d_model):
        """Create positional encoding matrix of given length"""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle both even and odd d_model dimensions
        # For even indices: use all div_term values
        # For odd indices: use div_term[:-1] to match the number of odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # When d_model is odd, we have one less odd index than even index
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        return pe
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # If sequence length exceeds current positional encoding, extend it
        if seq_len > self.pe.size(1):
            # Calculate new max_len (round up to next multiple of 1000 for efficiency)
            new_max_len = ((seq_len // 1000) + 1) * 1000
            # Create extended positional encoding
            extended_pe = self._create_pe(new_max_len, self.d_model)
            extended_pe = extended_pe.unsqueeze(0)  # (1, new_max_len, d_model)
            # Update the buffer
            self.pe = extended_pe.to(x.device)
            self.max_len = new_max_len
        
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout_ff=0.1, dropout_attn=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_ff = dropout_ff
        self.dropout_attn = dropout_attn
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout_attn,
            batch_first=True
        )
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_ff),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout_ff)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-norm: Self-attention with residual connection
        # Apply layer norm before attention (Pre-norm is more stable)
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Pre-norm: Feedforward with residual connection
        # Apply layer norm before feedforward
        ff_out = self.feedforward(self.norm2(x))
        x = x + ff_out
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder backbone for PA/DPD models"""
    
    def __init__(self, input_size, output_size, num_layers, d_model, 
                n_heads=8, d_ff=2048, dropout_ff=0.1, dropout_attn=0.1, bias=True,
                use_mlp_embedding=True):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_ff = dropout_ff
        self.dropout_attn = dropout_attn
        self.bias = bias
        self.use_mlp_embedding = use_mlp_embedding
        
        # Input projection/embedding layer
        # For continuous values (I, Q), this is more of a "projection" than "embedding"
        # but we keep the name for consistency with Transformer terminology
        if use_mlp_embedding:
            # More expressive MLP-based embedding (better for complex patterns)
            embedding_dim = max(d_model // 2, input_size * 4)
            self.input_embedding = nn.Sequential(
                nn.Linear(input_size, embedding_dim, bias=bias),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_ff),
                nn.Linear(embedding_dim, d_model, bias=bias)
            )
        else:
            # Simple linear projection (efficient, standard approach)
            self.input_embedding = nn.Linear(input_size, d_model, bias=bias)
        
        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_ff, dropout_attn)
            for _ in range(num_layers)
        ])
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, output_size, bias=bias)
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize input embedding
        if isinstance(self.input_embedding, nn.Sequential):
            # MLP-based embedding
            for module in self.input_embedding:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        else:
            # Simple linear embedding
            nn.init.xavier_uniform_(self.input_embedding.weight)
            if self.input_embedding.bias is not None:
                nn.init.constant_(self.input_embedding.bias, 0)
        
        # Initialize encoder layers
        for layer in self.encoder_layers:
            # Initialize self-attention weights
            for name, param in layer.self_attn.named_parameters():
                if 'weight' in name:
                    if 'in_proj' in name:
                        # Combined q, k, v projection
                        nn.init.xavier_uniform_(param)
                    elif 'out_proj' in name:
                        nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            
            # Initialize feedforward weights
            for module in layer.feedforward:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            # Layer norm is initialized to 1 for scale and 0 for bias by default
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.constant_(self.output_projection.bias, 0)
    
    def forward(self, x, h_0=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state (ignored for compatibility, but kept for interface)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
        """
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout to embeddings
        x = F.dropout(x, p=self.dropout_ff, training=self.training)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Output projection
        out = self.output_projection(x)  # (batch_size, seq_len, output_size)
        
        return out

