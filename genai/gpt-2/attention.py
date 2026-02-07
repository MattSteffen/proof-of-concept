import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention as used in GPT-2"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Combined Q, K, V projections (more efficient than separate)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V all at once
        qkv = self.W_qkv(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len, seq_len)
        
        # Causal mask
        if mask is None:
            mask = self._create_causal_mask(seq_len, x.device)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # Broadcast to all heads/batches
        
        # Softmax
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (batch, n_heads, seq_len, d_k)
        
        # Reshape: concatenate heads
        out = out.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_k)
        out = out.reshape(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        # Final projection with residual dropout
        return self.resid_dropout(self.W_o(out))
    
    @staticmethod
    def _create_causal_mask(seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))