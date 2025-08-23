import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from tqdm import tqdm
import pickle
import os

# Simple tokenizer for demonstration
class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        return torch.tensor([self.char_to_idx.get(char, 0) for char in text])
    
    def decode(self, indices):
        return ''.join([self.idx_to_char.get(idx.item(), '') for idx in indices])

# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_texts = []
        
        for text in texts:
            if len(text) > 0:
                encoded = self.tokenizer.encode(text)
                if len(encoded) > 0:
                    self.encoded_texts.append(encoded)
    
    def __len__(self):
        return len(self.encoded_texts)
    
    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        if len(encoded) > self.max_length:
            start_idx = random.randint(0, len(encoded) - self.max_length)
            encoded = encoded[start_idx:start_idx + self.max_length]
        
        # Pad if necessary
        if len(encoded) < self.max_length:
            padding = torch.zeros(self.max_length - len(encoded), dtype=torch.long)
            encoded = torch.cat([encoded, padding])
        
        return encoded

# Text diffusion model
class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_steps=1000):
        super().__init__()
        self.n_steps = n_steps
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Simple transformer-like architecture
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Linear beta schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x, t):
        # Convert indexes to embeddings
        x_embed = self.embedding(x)
        
        # Get noise level
        a_bar = self.alpha_bar[t]
        
        # Sample noise
        noise = torch.randn_like(x_embed)
        
        # Add noise
        noisy_x = torch.sqrt(a_bar).view(-1, 1, 1) * x_embed + torch.sqrt(1 - a_bar).view(-1, 1, 1) * noise
        
        return noisy_x, noise
    
    def forward(self, x, t):
        # Add noise to input
        noisy_x, target_noise = self.add_noise(x, t)
        
        # Time embedding
        t_embed = self.time_embed(t.float().view(-1, 1))
        
        # Encode noisy input
        encoded = self.encoder(noisy_x)
        
        # Combine encoded input with time embedding
        t_embed = t_embed.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        combined = torch.cat([encoded, t_embed], dim=2)
        
        # Predict noise
        pred_noise = self.decoder(combined)
        
        return pred_noise, target_noise
    
    def sample(self, prompt_tokens, max_length, device, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Start with random data
            x = torch.randint(0, self.embedding.num_embeddings, (1, max_length), device=device)
            
            # Copy prompt if provided
            if prompt_tokens is not None and len(prompt_tokens) > 0:
                prompt_len = min(len(prompt_tokens), max_length)
                x[0, :prompt_len] = prompt_tokens[:prompt_len]
            
            # Reverse diffusion process
            for t in tqdm(range(self.n_steps - 1, -1, -1), desc="Sampling"):
                t_tensor = torch.tensor([t], device=device)
                
                # Get embedding
                x_embed = self.embedding(x)
                
                # Time embedding
                t_embed = self.time_embed(t_tensor.float().view(-1, 1))
                
                # Encode
                encoded = self.encoder(x_embed)
                
                # Combine encoded with time embedding
                t_embed = t_embed.unsqueeze(1).expand(-1, encoded.shape[1], -1)
                combined = torch.cat([encoded, t_embed], dim=2)
                
                # Predict logits
                logits = self.decoder(combined)
                
                # Apply temperature
                logits = logits / temperature
                
                # Sample from logits
                probs = F.softmax(logits, dim=-1)
                x = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, -1)
            
        return x

# Training function
def train_model(model, train_data, epochs, device, learning_rate=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, model.n_steps, (batch.shape[0],), device=device)
            
            # Forward pass
            pred_logits, _ = model(batch, t)
            
            # Compute loss
            loss = criterion(pred_logits.view(-1, pred_logits.shape[-1]), batch.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_data):.4f}")
    
    return model

# Save tokenizer separately to avoid PyTorch serialization issues
def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Simple Text Diffusion Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--sample', action='store_true', help='Sample from the model')
    parser.add_argument('--prompt', type=str, default='', help='Prompt for sampling')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model_path', type=str, default='text_diffusion_model.pt', help='Model path')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.pkl', help='Tokenizer path')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension size')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data file')
    args = parser.parse_args()
    
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sample training data (if no data path provided)
    sample_texts = [
        "Text diffusion models are a type of generative model.",
        "They work by adding noise to text and then learning to denoise.",
        "This is a very simplified version for demonstration purposes.",
        "The model generates text by gradually denoising random data.",
        "Machine learning is a field of artificial intelligence.",
        "Natural language processing deals with text data.",
        "Deep learning models can learn complex patterns.",
        "Transformers have revolutionized NLP tasks.",
        "Python is a popular programming language for AI."
    ]
    
    # Load data from file if provided
    if args.data_path:
        try:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                sample_texts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(sample_texts)} lines from {args.data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using default sample texts")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.fit(sample_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = TextDataset(sample_texts, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create or load model
    if args.train:
        model = TextDiffusionModel(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim
        )
        
        # Train the model
        model = train_model(model, dataloader, args.epochs, device)
        
        # Save the model (weights only) and tokenizer separately
        torch.save(model.state_dict(), args.model_path)
        save_tokenizer(tokenizer, args.tokenizer_path)
        print(f"Model saved to {args.model_path}")
        print(f"Tokenizer saved to {args.tokenizer_path}")
    
    # Sample from the model
    if args.sample:
        try:
            # Load the tokenizer and model separately
            tokenizer = load_tokenizer(args.tokenizer_path)
            
            model = TextDiffusionModel(
                vocab_size=tokenizer.vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim
            )
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            
            # Encode prompt
            prompt_tokens = tokenizer.encode(args.prompt) if args.prompt else None
            if prompt_tokens is not None:
                prompt_tokens = prompt_tokens.to(device)
            
            # Sample from the model
            print(f"Generating with prompt: '{args.prompt}'")
            sampled_tokens = model.sample(
                prompt_tokens=prompt_tokens,
                max_length=args.max_length,
                device=device,
                temperature=args.temperature
            )
            
            # Decode and print
            generated_text = tokenizer.decode(sampled_tokens[0])
            print(f"Generated text: {generated_text}")
            
        except Exception as e:
            print(f"Error sampling from model: {e}")
            print("Make sure you've trained the model first.")
    
    # Interactive CLI if neither train nor sample is specified
    if not args.train and not args.sample:
        try:
            # Load the tokenizer and model separately
            tokenizer = load_tokenizer(args.tokenizer_path)
            
            model = TextDiffusionModel(
                vocab_size=tokenizer.vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim
            )
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            
            print("Interactive mode. Type 'exit' to quit.")
            while True:
                prompt = input("Enter prompt: ")
                if prompt.lower() == 'exit':
                    break
                
                # Encode prompt
                prompt_tokens = tokenizer.encode(prompt)
                prompt_tokens = prompt_tokens.to(device)
                
                # Sample from the model
                sampled_tokens = model.sample(
                    prompt_tokens=prompt_tokens,
                    max_length=args.max_length,
                    device=device,
                    temperature=args.temperature
                )
                
                # Decode and print
                generated_text = tokenizer.decode(sampled_tokens[0])
                print(f"Generated: {generated_text}")
                
        except FileNotFoundError:
            print("Error: Model or tokenizer file not found.")
            print("Make sure you've trained the model first.")
        except Exception as e:
            print(f"Error in interactive mode: {e}")
            print("Make sure you've trained the model first.")

if __name__ == "__main__":
    main()