# train.py
# Heavier example; for a quick smoke test use sample.py.
import torch
import fire
from torch.utils.data import DataLoader
from tokenizer import GPT2Tokenizer
from gpt2 import GPT2, GPT2Trainer, TextDataset
from sample_data import sample_texts
from eval import TextGenerator

DEFAULT_CONFIG = {
        'vocab_size': 50257,
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'max_seq_len': 1024,
        'dropout': 0.1,
        'lr': 3e-4,
        'weight_decay': 0.1,
        'warmup_steps': 2000,
        'max_steps': 100000,
        'grad_clip': 1.0,
        'batch_size': 8,
        'seq_len': 512,
        'epochs': 1,
        'eval_every': 500,
        'save_path': 'gpt-2/gpt2_model.pt',
        'gen_prompt': 'The meaning of life is',
        'gen_max_new_tokens': 100,
        'gen_temperature': 1.0,
        'gen_top_k': 50,
}


def main(**overrides):
    # Configuration (overridable via Fire CLI)
    config = DEFAULT_CONFIG.copy()
    for key, value in overrides.items():
        if key not in config:
            raise ValueError(f"Unknown config option: {key}")
        if value is not None:
            config[key] = value
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer()
    
    # Load and tokenize data
    text = "\n".join(sample_texts)
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")
    
    if len(tokens) < 2:
        raise ValueError("Not enough tokens to create training sequences.")

    # Split into train/val
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Ensure we have enough training data to form at least one sequence
    if len(train_tokens) < 2:
        train_tokens = tokens
        val_tokens = []

    # Adjust sequence length based on available data
    seq_len = min(config['seq_len'], len(train_tokens) - 1)
    if seq_len < 1:
        raise ValueError("Not enough tokens to set a valid sequence length.")
    if seq_len != config['seq_len']:
        print(f"Adjusting seq_len to {seq_len} based on data size.")
    config['seq_len'] = seq_len
    
    # Create datasets
    train_dataset = TextDataset(train_tokens, config['seq_len'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    val_loader = None
    if len(val_tokens) >= 2:
        val_dataset = TextDataset(val_tokens, config['seq_len'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = GPT2(config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Initialize trainer
    trainer = GPT2Trainer(model, config, device)
    
    # Train
    trainer.train(
        train_loader,
        epochs=config['epochs'],
        eval_every=config['eval_every'],
        eval_fn=(lambda: trainer.evaluate(val_loader)) if val_loader else None
    )
    
    # Save model
    torch.save(model.state_dict(), config['save_path'])
    
    # Test generation
    generator = TextGenerator(model, tokenizer, device)
    output = generator.generate(
        config['gen_prompt'],
        max_new_tokens=config['gen_max_new_tokens'],
        temperature=config['gen_temperature'],
        top_k=config['gen_top_k'],
    )
    print(f"\nGenerated text:\n{output}")

if __name__ == "__main__":
    fire.Fire(main)
