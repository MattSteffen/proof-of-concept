import torch
import torch.nn as nn
import math
from embeddings import GPT2Encoder
from modules import TransformerBlock

class GPT2Config:
    """GPT-2 model configurations"""
    
    # GPT-2 Small (124M params)
    SMALL = {
        'vocab_size': 50257,
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,       # 4 * d_model
        'max_seq_len': 1024,
        'dropout': 0.1,
    }
    
    # GPT-2 Medium (355M params)
    MEDIUM = {
        'vocab_size': 50257,
        'd_model': 1024,
        'n_heads': 16,
        'n_layers': 24,
        'd_ff': 4096,
        'max_seq_len': 1024,
        'dropout': 0.1,
    }


class GPT2(nn.Module):
    """Complete GPT-2 model"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.embeddings = GPT2Encoder(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )
            for _ in range(config['n_layers'])
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(config['d_model'])
        
        # Output projection (tied to input embeddings)
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Weight tying: share embeddings and output projection
        self.lm_head.weight = self.embeddings.token_embedding.weight
        
        # Apply special initialization
        self.apply(self._init_weights)
        self._init_residual_projections()
    
    def _init_weights(self, module):
        """GPT-2 weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def _init_residual_projections(self) -> None:
        """Scale residual projections to stabilize deep transformer training."""
        n_layers = self.config['n_layers']
        scaled_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.W_o.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.ffn.linear2.weight, mean=0.0, std=scaled_std)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embeddings
        x = self.embeddings(input_ids)
        
        # Create causal mask once
        seq_len = input_ids.shape[1]
        mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    @staticmethod
    def _create_causal_mask(seq_len: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: (batch, seq_len) starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
        Returns:
            (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.config['max_seq_len']:]
            
            # Get predictions
            logits = self(idx_cond)
            
            # Only look at last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TextDataset(Dataset):
    """Dataset for text data"""
    
    def __init__(self, tokens: list[int], seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
    
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


class GPT2Trainer:
    """Training loop for GPT-2"""
    
    def __init__(self, model: GPT2, config: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # AdamW optimizer (Adam with weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 3e-4),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.get('weight_decay', 0.1)
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler(
            config.get('warmup_steps', 2000),
            config.get('max_steps', 100000)
        )
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 1.0)
    
    def _create_scheduler(self, warmup_steps: int, max_steps: int):
        """Cosine learning rate schedule with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            progress = min(progress, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Single training step"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        logits = self.model(inputs)
        
        # Cross-entropy loss
        # Reshape for cross_entropy: (batch * seq_len, vocab_size)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train(self, dataloader: DataLoader, epochs: int, 
              eval_every: int = 500, eval_fn=None):
        """Full training loop"""
        step = 0
        
        for epoch in range(epochs):
            epoch_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
            )
            for batch in epoch_bar:
                loss = self.train_step(batch)
                step += 1
                
                lr = self.optimizer.param_groups[0]['lr']
                epoch_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", step=step)
                
                if step % 100 == 0:
                    print(f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}")
                
                if eval_fn and step % eval_every == 0:
                    eval_loss = eval_fn()
                    print(f"  Eval Loss: {eval_loss:.4f}")
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            logits = self.model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        self.model.train()
        return total_loss / n_batches
