# config for training GPT-2 (124M), loss comes around to be ~2.85 which surpasses openAI's GPT-2 loss
# command to train:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# total tokens = 300 billion
max_iters = 600000
lr_decay_iters = 600000

# evaluation params
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
