import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # independent data sequences that will be processed in parallel
block_size = 8 # context length for predictions (first block_size - 1 predictions will have smaller block_size due to lack of data)
max_iters = 5000 # maxiumum iterations for the neural network
eval_interval = 500 # number of iterations after which loss estimation will be made
eval_iters = 200 # number of iterations for loss estimation
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32

torch.manual_seed(1337) # seed = encode('gpt') ;)

# input text data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    input = f.read()

vocab = sorted(list(set(input)))
vocab_size = len(vocab)

# create char -> int encoder and int -> char decoder
char_to_int = {ch:i for i,ch in enumerate(vocab)}
int_to_char = {i:ch for i,ch in enumerate(vocab)}
encode = lambda str: [char_to_int[c] for c in str] # encodes a string into a list of integers
decode = lambda ls: ''.join([int_to_char[i] for i in ls]) # decodes a list of integers into a string

# split train and test data (90% split)
data = torch.tensor(encode(input), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# data loading into batches
def get_batch(split):
    # generate a batch for data for inputs (x) and targets (y) 
    data = train_data if split == "train" else validation_data
    indeces = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in indeces])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indeces])
    x, y = x.to(device), y.to(device) # move data to device
    return x, y

# caculate weighted average loss accross batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one self-attention head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores for each token
        w = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # tokens only have knowledge of the past tokens, not future token
        w = F.softmax(w, dim=-1)
        # weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = w @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention together """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(Head(head_size))
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class FeedForward(nn.Module):
    """ a linear layer followed by a non-linearity (ReLU) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)

# initial Bigram language model
class BigramLanguageModel(nn.Module):
    """ model definition """
    def __init__(self):
        super().__init__()
        # initialize an embedding table for every token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # initialize a positional embedding table for every token
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        # 4 heads of 8-dimensional self-attention, post-concat becomes 32 dimentional
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets = None):
        B, T = index.shape
        # index.shape == targets.shape == (B, T)
        token_embds = self.token_embedding_table(index) # (B, T, C)
        positional_embds = self.positional_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embds + positional_embds # (B, T, C)
        x = self.sa_heads(x) # apply one self-attention head
        x = self.ffwd(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # PyTorch expects channels to be the first dimension for cross entropy so changing shapes
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index.shape = (B, T)
        for _ in range(max_new_tokens):
            # crop index to last block
            index_cropped = index[:, -block_size:]
            # get predictions
            logits, loss = self(index_cropped)
            # reduce to the last time dimension
            logits = logits[:, -1, :] # (B, C)
            # get probabilities by applying softmax
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, next_index), dim=1) # (B, T+1)
        return index

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    # every once in eval_interval iterations, calculate the average loss accross all batches on train and validation sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"iter {i}: train loss {losses['train']:.4f}, validation loss {losses['validation']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    # backward pass
    loss.backward()
    # optimizer update
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
