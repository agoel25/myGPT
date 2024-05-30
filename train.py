# Training script for the model
# Can run on a CPU, a GPU or (hopefully) in a distributed data parallel (ddp) environment
# 
# NOTE: The text is assumed to be pre-tokenized

import os
import time
import math

import numpy as np
import torch
import wandb

from model import GPTConfig, GPT

# default config values to train gpt2 on OpenWebText, change values for other dataset if required
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval

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