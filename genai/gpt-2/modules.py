import torch
import torch.nn as nn
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.resid_dropout = nn.Dropout(dropout)
        
        # GELU approximation used in GPT-2
        self.gelu = nn.GELU(approximate='tanh')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.linear1(x)      # (batch, seq_len, d_ff)
        x = self.gelu(x)          # Element-wise GELU
        x = self.linear2(x)       # (batch, seq_len, d_model)
        return self.resid_dropout(x)


class TransformerBlock(nn.Module):
    """Single transformer decoder block"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Pre-norm architecture (GPT-2 style)
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Attention with residual
        x = x + self.attention(self.ln1(x), mask)
        
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x