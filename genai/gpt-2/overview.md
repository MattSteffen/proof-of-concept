# GPT-2 From Scratch Overview

This folder contains a minimal GPT-2-style implementation in PyTorch, plus a small
training/evaluation harness and tokenizer wrapper.

## Files

- `attention.py`: Multi-head causal self-attention module with residual dropout.
- `embeddings.py`: Token and positional embeddings with embedding dropout.
- `eval.py`: Text generation utilities and perplexity computation.
- `gpt2.py`: Core GPT-2 model, training loop, dataset, and initialization logic.
- `main.py`: Example training script using sample data and configuration.
- `modules.py`: Transformer block and feed-forward network components.
- `overview.md`: This overview of the folder contents.
- `sample.py`: One-shot tiny train + eval + generation smoke test (run via `uv run gpt-2/sample.py`).
- `sample_data.py`: Small toy corpus used for quick smoke tests.
- `tokenizer.py`: GPT-2 tokenizer wrapper around tiktoken.
