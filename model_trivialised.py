"""
Trivialised GPT model definition, same as model.py but with no optimization code for better readability.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    # multi-headed attention module
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
        output = self.c_proj(output)
        return output

class MLP(nn.Module):
    def __init__(self, config):
        # just a simple multi layer perceptron with 2 linear layers and a GELU non-linearity
        super().__init__()
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
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # attention handles the communication between tokens, feed forward handles the computation

        # since gradients are equally distributed for addition during backprop, same gradients will flow through the 
        # residual pathway and the the blocks
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
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of blocks, one per hidden layer
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # projection from n_embd back to vocab size for final prediction

        # weight sharing between wte and lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # initialize parameters
        self.apply(self._init_weights)
    
    # default initialize update according to gpt2 paper
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        B, T = index.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=index.device)
        positional_emd = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(index)
        x = token_emb + positional_emd
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits are one softmax away from being probabilities
        loss = None
        if targets is not None:
            # cross entropy only likes 2 dimentional dims so we flatten out our tensors
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss