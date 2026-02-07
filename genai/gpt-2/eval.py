import math
import torch
import torch.nn.functional as F
from gpt2 import GPT2
from tokenizer import GPT2Tokenizer
from torch.utils.data import DataLoader as PyTorchDataLoader

class TextGenerator:
    """Text generation with various sampling strategies"""
    
    def __init__(self, model: GPT2, tokenizer: GPT2Tokenizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50, 
                 top_p: float = 0.9) -> str:
        """
        Generate text with top-k and top-p (nucleus) sampling.
        
        Args:
            prompt: starting text
            temperature: higher = more random
            top_k: only sample from top k tokens
            top_p: nucleus sampling threshold
        """
        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        for _ in range(max_new_tokens):
            # Crop to context window
            context = input_ids[:, -self.model.config['max_seq_len']:]
            
            # Get logits
            logits = self.model(context)[:, -1, :]  # Last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(input_ids[0].tolist())


def compute_perplexity(model: GPT2, dataloader: PyTorchDataLoader, device: str) -> float:
    """Compute perplexity on a dataset"""
    model.eval()
    total_nll = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            # Negative log likelihood
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction='sum'
            )
            
            total_nll += nll.item()
            total_tokens += targets.numel()
    
    model.train()
    return math.exp(total_nll / total_tokens)