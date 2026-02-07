import torch
import torch.nn as nn
import math

class GPT2Encoder(nn.Module):
    """Token + Positional embeddings for GPT-2"""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        
        # Token embeddings: vocab_size × d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings: max_seq_len × d_model
        # GPT-2 uses learned positional embeddings (not sinusoidal)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights (GPT-2 uses specific initialization)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer token IDs
        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding(token_ids)  # (batch, seq_len, d_model)
        
        # Get position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions)  # (seq_len, d_model)
        
        # Add them together
        return self.dropout(tok_emb + pos_emb)  # Broadcasting adds pos_emb to each batch