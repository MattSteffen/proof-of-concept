# tokenizer.py
import tiktoken  # OpenAI's tokenizer library

class GPT2Tokenizer:
    """Wrapper around tiktoken for GPT-2"""
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = 50257  # GPT-2 vocabulary size
    
    def encode(self, text: str) -> list[int]:
        """Text -> token IDs"""
        return self.encoder.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        """Token IDs -> text"""
        return self.encoder.decode(tokens)