from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # masking as done by openAI
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, input):
        B, T, C = input.size() # B = batch size, T = context length, C = m_embd (embedding dimensionality)
        q, k, v  = self.c_attn(input).split(self.n_embd, dim=2) # calculate query, key and value for all heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        w = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        w = w.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # makes sure context for a token is just the tokens before it
        w = F.softmax(w, dim=-1) # normalizes the attention
        output = w @ v # back to original shape
        output = output.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        output = self.residual_dropout(self.c_proj(output))
        return output

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