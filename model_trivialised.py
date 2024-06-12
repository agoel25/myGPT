from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # using tanh as approximator since GPT2 used it 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, input):
        input = self.c_fc(input)
        input = self.gelu(input)
        input = self.c_proj(input)
        return input

class Block(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # GPT-2 vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of blocks, one per hidden layer
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.head = nn.Linear(config.n_head, config.vocab_size, bias=False)