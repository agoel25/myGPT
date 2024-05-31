# Training script for the model
# Can run on a CPU, a GPU or (hopefully) in a distributed data parallel (ddp) environment
# 
# NOTE: The text is assumed to be pre-tokenized

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT

# ----------------------------------------------------------------------------------------------------------------------------------- #
# default config values designed to train the model on the OpenWebText dataset, change if running on some other dataset if required
out_dir = 'out'
eval_interval = 2000
eval_iters = 200
log_interval = 1
eval_only = False # if true, exit after the first iteration itself
init_from = 'start' # start, resume or gpt2 (for pretrained checkpoints)

gradient_accumulation_steps = 5 * 8 # we want to accumulate gradients accross mini batches before the backward pass
batch_size = 12
block_size = 1024

device = 'cuda' # 'cpu', 'cuda' or (for macbooks) 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# ----------------------------------------------------------------------------------------------------------------------------------- #

# environment and i/o setup
# NOTE: ddp is a method in PyTorch for parallelizing model training accress multiple GPUs and nodes, each setup has a rank
ddp = int(os.environ.get('RANK', -1)) != -1 # get the RANK env variable to see if this is a ddp environment 
if ddp:
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
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
