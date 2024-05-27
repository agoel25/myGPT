import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # independent data sequences that will be processed in parallel
block_size = 8 # context length for predictions (first block_size - 1 predictions will have smaller block_size due to lack of data)
max_iters = 3000 # maxiumum iterations for the neural network
eval_iters = 200 # number of iterations for loss estimation
eval_interval = 300 # number of iterations after which loss estimation will be made
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(455458) # seed = encode('gpt') ;)

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

# caculate weighted average loss
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

# initial Bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # initialize an embedding table for every token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets = None):

        # index.shape == targets.shape == (B, T)
        logits = self.token_embedding_table(index) # (B, T, C)

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
            # get predictions
            logits, loss = self(index)
            # reduce to the last time dimension
            logits = logits[:, -1, :] # (B, C)
            # get probabilities by applying softmax
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, next_index), dim=1) # (B, T+1)
        return index

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):

    # every once in eval_interval iterations, calculate the loss on train and validation sets
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    # backward pass for the NN
    loss.backward()
    # optimizer update step for the NN
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
