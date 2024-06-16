import tiktoken
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from model_trivialised import GPT, GPTConfig

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")

        self.current_position = 0 
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T
        # if next batch is out of bounds, reset the position
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------------------

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

train_loader = DataLoaderLite(B=4, T=32)

# use the tf32 precision 
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304)) # overriding vocab size to the nearest number divisible by 2
model.to(device)
if torch.cuda.is_available():
    model = torch.compile(model) # compiles the NN, we pay in compilation time for better runtime

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    if torch.cuda.is_available():
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # using automatic mixed precision for better performance on gpus
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # set maximum gradient to 1.0
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize() # ensures that gpu has finished processing before continuing
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"step {i}, loss {loss.item():.6f}, norm: {norm:.4f}, time {dt:.2f}ms")

import sys
sys.exit(0)


# prefix tokens
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

num_return_sequences = 5
max_length = 30

# generate
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # only use the logits in the last position
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # only keep the top 50 probabilities, remove lower ones to avoid model from getting sidetracked
        # 50 is the default used by huggingface's pipeline so we use this too
        topk_probs, topk_indeces = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indeces, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("> " + decoded)