import itertools
import time
import torch
from torch.utils.data import DataLoader

from tokenizer import GPT2Tokenizer
from gpt2 import GPT2, GPT2Trainer, TextDataset
from sample_data import sample_texts
from eval import TextGenerator, compute_perplexity


def _choose_split(tokens_len: int, seq_len: int) -> int | None:
    min_split = seq_len + 1
    max_split = tokens_len - (seq_len + 1)
    if max_split < min_split:
        return None
    split = int(0.9 * tokens_len)
    split = max(min(split, max_split), min_split)
    return split


def _prepare_splits(tokens: list[int], desired_seq_len: int, min_seq_len: int) -> tuple[int, list[int], list[int]]:
    tokens_len = len(tokens)
    if tokens_len <= min_seq_len + 1:
        raise ValueError(
            "Not enough tokens for a train/val split. "
            f"Need > {min_seq_len + 1} tokens, got {tokens_len}."
        )

    seq_len = min(desired_seq_len, tokens_len - 2)
    seq_len = min(seq_len, max(min_seq_len, seq_len))

    while seq_len >= min_seq_len:
        split = _choose_split(tokens_len, seq_len)
        if split is not None:
            train_tokens = tokens[:split]
            val_tokens = tokens[split:]
            if len(train_tokens) > seq_len and len(val_tokens) > seq_len:
                return seq_len, train_tokens, val_tokens
        seq_len = seq_len // 2 if seq_len > min_seq_len else seq_len - 1

    raise ValueError(
        "Unable to find a valid sequence length for the train/val split. "
        f"Tokens: {tokens_len}, min_seq_len: {min_seq_len}."
    )


def main() -> None:
    torch.manual_seed(0)

    config = {
        "vocab_size": 50257,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "max_seq_len": 64,
        "dropout": 0.1,
        "lr": 3e-4,
        "weight_decay": 0.1,
        "warmup_steps": 20,
        "max_steps": 100,
        "grad_clip": 1.0,
        "batch_size": 4,
        "seq_len": 32,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer()
    text = "\n".join(sample_texts)
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens)}")

    desired_seq_len = min(config["seq_len"], config["max_seq_len"])
    seq_len, train_tokens, val_tokens = _prepare_splits(tokens, desired_seq_len, min_seq_len=8)

    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Train/val datasets must each contain at least one sequence. "
            f"train_sequences={len(train_dataset)}, val_sequences={len(val_dataset)}."
        )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = GPT2(config)
    print(f"Model parameters: {model.count_parameters():,}")

    trainer = GPT2Trainer(model, config, device)

    train_steps = 25
    log_every = 5
    losses: list[float] = []
    start_time = time.time()

    data_iter = itertools.cycle(train_loader)
    for step in range(1, train_steps + 1):
        batch = next(data_iter)
        loss = trainer.train_step(batch)
        losses.append(loss)

        if step == 1 or step % log_every == 0 or step == train_steps:
            lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"Step {step:>2}/{train_steps} | Loss: {loss:.4f} | LR: {lr:.2e}")

    elapsed = time.time() - start_time
    val_loss = trainer.evaluate(val_loader)
    val_ppl = compute_perplexity(model, val_loader, device)

    avg_recent_loss = sum(losses[-min(5, len(losses)):]) / min(5, len(losses))
    print("\nRun summary")
    print(f"- train_tokens: {len(train_tokens)} | val_tokens: {len(val_tokens)}")
    print(f"- seq_len: {seq_len} | batch_size: {config['batch_size']}")
    print(f"- train_sequences: {len(train_dataset)} | val_sequences: {len(val_dataset)}")
    print(f"- steps: {train_steps} | avg_recent_loss: {avg_recent_loss:.4f}")
    print(f"- val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.2f}")
    print(f"- elapsed_sec: {elapsed:.2f}")

    generator = TextGenerator(model, tokenizer, device)
    output = generator.generate("The meaning of life is", max_new_tokens=60, temperature=1.0, top_k=50, top_p=0.9)
    print("\nGenerated text:")
    print(output)


if __name__ == "__main__":
    main()
