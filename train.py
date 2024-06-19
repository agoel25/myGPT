# Training script for the model
# Can run on a CPU, a GPU or (hopefully) in a distributed data parallel (ddp) environment
# 
# NOTE: The input text is assumed to be pre-tokenized

import os
from contextlib import nullcontext
import pickle
import math
import time

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

from model import GPTConfig, GPT

# default config values designed to train the model on the OpenWebText dataset, change if running on some other dataset if required
# ----------------------------------------------------------------------------------------------------------------------------------- #
# housekeeping related
out_dir = 'out'
eval_interval = 2000
eval_iters = 200
log_interval = 1
eval_only = False # exit after the first iteration itself
init_from = 'start' # start, resume or gpt2 (for pretrained checkpoints)
always_save_checkpoint = True # always save a checkpoint after each evaluation

# data related
dataset = 'openwebtext' # name of data directory in ./data folder to be used for training
gradient_accumulation_steps = 5 * 8 # we want to accumulate gradients across mini batches to simulate larger batch size
batch_size = 12
block_size = 1024

# model related
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# pytorch related
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps' # for macbooks
print(f"Using device {device}")
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
backend = 'nccl' # distributed data parallel settings, example: nccl, gloo

# optimizer related
learning_rate = 6e-4 # initial learning rate
max_iters = 600000 # total number of iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate related
decay_lr = True # decay the learning rate as iterations progress
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = learning_rate * 0.1

# weight and bias logging related
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# ----------------------------------------------------------------------------------------------------------------------------------- #

# setup config 
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# environment and i/o setup
# DDP (Distributed Data Parallel) is a method in PyTorch for parallelizing model training across multiple GPUs
# torchrun command sets environment variables RANK, LOCAL_RANK and WORLD_SIZE
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
torch.manual_seed(2506 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
# setting up a context for automating mixed precision for better performance and memory management when using GPUs
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] # map data types to their corresponding pytorch data types
context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load the data, that is inputs (xs) and targets (ys)
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # using same memory mapped arrays as model.py for better performance
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    index = torch.randint(len(data) - block_size, (batch_size,))
    # get input and target tensors for the randomly selected indices from across the data
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
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
    state_dict = checkpoint['model']
    # checkpoints state dictionaries sometimes get an unwanted prefix, we have to remove it for model.py to work as expected
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

# crop down the model block size if required
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# initializing GradScaler to scale gradients for mixed (half + single) precision training; uses both 16-bit and 32-bit floating points 
# to reduce memory usage and accelerate training since small gradients underflow and fall to 0 during back prop
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# initialize the optimizer (AdamW)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer']) # use state dictionary from checkpoint if resuming training
checkpoint = None

# compile the model (if available)
if compile:
    print("Compiling the model... (hang tight, takes round a minute)")
    unoptimized_model = model
    # compile is a kernel fusion operation, reducing time spent sending data between memory and processing units
    model = torch.compile(model)

# create a DistributedDataParallel object (if available)
if ddp:
    model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

# estimate loss using many batches since individual batches can have very luck or very unlucky data
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # put model in evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with context:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # put model back in train mode
    return out

# learning rate scheduler
def get_lr(iter):
    # if iter < warmup_iters, learning rate increases linearly from 0 to learning_rate (this is a linear warmup)
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # if iter > lr_decay_iters, learning rate has a constant value = min_lr
    if iter > lr_decay_iters:
        return min_lr
    # between warmup_iters and lr_decay_iters, use cosine smooth decay to go down from learning_rate to min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters) # calculate progress between warmup_iters and lr_decay_iters
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # use cosine function to calculate learning rate coefficient (0 <= coeff <= 1)
    return min_lr + coeff * (learning_rate - min_lr) # return final learning rate

# weights and biases logging
if wandb_log and master_process: # only master process handles the logging
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# finally, after 250 lines of code, we are ready for training :)
# training loop
X, Y = get_batch('train') # first batch
t0 = time.time() # for logging
raw_model = model.module if ddp else model
while True:
    # fetch and set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # run if it has been eval_interval iterations since the last evaluation 
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Iteration {iter_num}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}")
        # logging
        if wandb_log: 
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
    # run if the current validation loss is the best so far or if a checkpoint needs to be saved
    if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0: # if the model just started, do not save checkpoint
                # create checkpoint dictionary and save it to out_dir
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    # early-exit for evaluation only mode 
    if iter_num == 0 and eval_only:
        break
    
    # starting gradient accumulation loop
    # gradients accumulate over multiple mini batches to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in ddp we synchronize the gradients at the last step
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        # forward pass and scaling the loss down to account for the gradient accumulation
        with context: # use mixed prevision operations if needed
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        # backward pass with loss scaling
        scaler.scale(loss).backward()
    # clip the gradient to prevent them from being too large -> stabilizes training
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler for next training iteration
    scaler.step(optimizer)
    scaler.update()
    # zero out the gradients
    optimizer.zero_grad(set_to_none=True)

    # logging with timing
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # master process handles logging, log if current iteration is a multiple of log_interval
    if iter_num % log_interval == 0 and master_process:
        # scaling up loss to to approximate total loss, since loss was scaled down during gradient accumulation
        loss_out = loss.item() * gradient_accumulation_steps
        print(f"Iter {iter_num}: Loss {loss_out:.4f}, Time {dt*1000:.2f}ms")
    iter_num += 1

    # termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()