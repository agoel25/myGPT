# Training script for the model
# Can run both on a single GPU and in a distributed data parallel (ddp) environment

import numpy as np
import torch

model = GPT()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    # every once in eval_interval iterations, calculate the average loss accross all batches on train and validation sets
    if i % eval_interval == 0 or iter == max_iters - 1:
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
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))