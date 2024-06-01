# Training script for the model
# Can run on a CPU, a GPU or (hopefully) in a distributed data parallel (ddp) environment
# 
# NOTE: The text is assumed to be pre-tokenized

import os
from contextlib import nullcontext
import pickle

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# default config values designed to train the model on the OpenWebText dataset, change if running on some other dataset if required
# ----------------------------------------------------------------------------------------------------------------------------------- #
# housekeeping related
out_dir = 'out'
eval_interval = 2000
eval_iters = 200
log_interval = 1
eval_only = False # if true, exit after the first iteration itself
init_from = 'start' # start, resume or gpt2 (for pretrained checkpoints)

# data related
dataset = 'openwebtext' # name of data directory in ./data folder to be used for training
gradient_accumulation_steps = 5 * 8 # we want to accumulate gradients accross mini batches before the backward pass
batch_size = 12
block_size = 1024

# model related
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# pytorch related
device = 'cuda' # 'cpu', 'cuda' or (for macbooks) 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
backend = 'nccl' # distributed data parallel settings, example: nccl, gloo
# ----------------------------------------------------------------------------------------------------------------------------------- #

# environment and i/o setup
# NOTE: ddp is a method in PyTorch for parallelizing model training accress multiple GPUs and nodes, each setup has a rank
ddp = int(os.environ.get('RANK', -1)) != -1 # get the RANK env variable to see if this is a ddp environment 
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # rank of the process on the local machine/node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of processes participating in the training
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    if ddp_rank == 0:
        master_process = True
    else:
        master_process = False
    seed_offset = ddp_rank # each process must have a a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0 
    gradient_accumulation_steps //= ddp_world_size # scale down the gradient accumulation iterations per process
else:
    # since we are not running on ddp, there is just one process 
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# housekeeping tasks
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
# setting up a context for automating mixed precision for better performance and memory management when using GPUs
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] # map data types to their corresponding pytorch data types
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load the data, that is inputs (xs) and targets (ys)
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # using same memory mapped arrays as model.py for better performance
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    index = torch.randint(len(data) - block_size, (batch_size,))
    # get input and target tensors for the randomly selected indeces from accross the data
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in index])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in index])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# variables for checkpoints to keep track of
iter_num = 0
best_val_loss = 1e9

# derive vocab_size from the metadata file
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size is {meta_vocab_size} at path {meta_path}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

# model initialization according to the 'init_from' variable
if init_from == 'start':
    print("Initializing the model from start")
    if meta_vocab_size is None:
        print("No specific vocab_size found, falling back to GPT-2's 50304")
        model_args['vocab_size'] = 50304
    else:
        model_args['vocab_size'] = meta_vocab_size
    # create the model
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig) # we have a new GPT-2 model built from scratch :)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # load mandatory model arguments from the checkpoint arguments
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconfif = GPTConfig(**model_args)
    model = GPT(gptconfif)
    state_dict = checkpoint['model']
    # checkpoints state dictionaries sometimes get an unwated prefix, we have to remove it for model.py to work as expected
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing the model from OpenAI's GPT-2 weights: {init_from}")
    # initialize gpt2 args
    override_args = dict(dropout=dropout) # override the dropout to user input
    # create the model
    model = GPT.from_pretrained(init_from, override_args)
    # maintain model_args dictionary 
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
