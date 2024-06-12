"""
GPT model definition
All parameter values are adopted from the config
Every class has 2 simple functions: constructor which initializes the layer/block and forward() which handles the forward pass
All naming conventions follow Hugging Face's transformer implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

For reference here is OpenAI's official GPT-2 TensorFlow implementation: https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from dataclasses import dataclass

@dataclass
class GPTConfig:
    """ Hyperparameters """
    vocab_size: int = 50257 # GPT-2 vocab size
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    """ Layer normalization with optional bias """
    def __init__(self, dims, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.bias = nn.Parameter(torch.zeros(dims)) if bias else None

    def forward(self, input):
        eps = 1e-5 # epsilon value taken from pytorch layer_norm documentation
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps)

class CausalSelfAttention(nn.Module):
    """ Multi-headed self attention (similar to my older commits but optimized as suggested by Andrej Kaparthy) """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # ensure embedding dimensionality is divisible by the number of attention heads
        # layer initializations
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # check if flash attention is available, if not, use bottom triangular matrix
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # use lower triangular matrix to ensure that knowledge only flows from the left, tokens after the current token are not included
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    
    def forward(self, input):
        B, T, C = input.size() # B = batch size, T = context length, C = m_embd (embedding dimensionality)
        q, k, v  = self.c_attn(input).split(self.n_embd, dim=2) # calculate query, key and value for all heads
        # swap n_head (dim = 2) and T (dim = 1) for regularization against other PyTorch methods
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Self-attention: (B, n_head, T, head_size) x (B, n_head, head_size, T) -> (B, n_head, T, T)
        if self.flash:
            # faster attention using Flash Attention CUDA kernels, note: no dropout during val
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # slower manual implementation of attention
            w = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            w = w.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # makes sure context for a token is just the tokens before it
            w = F.softmax(w, dim=-1) # normalizes the attention
            w = self.attn_dropout(w)
            output = w @ v # back to original shape
        output = output.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        output = self.residual_dropout(self.c_proj(output))
        return output

class MLP(nn.Module):
    """ Main multi-layer perceptron block """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # fully connected linear layeer
        self.gelu = nn.GELU(approximate='tanh') # gaussian error linear unit activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # projection back to original size
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer block: as defined in the 'attention is all you need' paper """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # attention handles the communication between tokens, feed forward handles the computation
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ Generative pre-trained transformer language model """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # transformer containing main components that make up GPT
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of positional embeddings
            drop = nn.Dropout(config.dropout), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of blocks, one per hidden layer
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # final layer norm 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # projection from n_embd back to vocab_size to predict final output
        self.transformer.wte.weight = self.lm_head.weight # tying input and output weights for better performance

        # initalize all weights
        self.apply(self.init_weights)
        # residual projections are specially initialized as per OpenAI's GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # print number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        """
        Helper function to return the number of parameters in the model
        Number of embedding weights gets deleted since we only want count for the model
        """
        n_params = sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()
        return n_params

    def init_weights(self, module):
        """
        Helper function to initialize all model weights
        Values:
        1. Gaussian distribution with a mean of 0 and standard deviation of 0.02
        2. If bias exists, all bias weights are 0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        device = index.device
        B, T = index.size()
        assert T <= self.config.block_size, f"Sequence length must be smaller than block size but {T} > {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # position index

        token_embds = self.transformer.wte(index) # (B, T, C), C = n_embd
        positional_embds = self.transformer.wpe(pos) # (T, C)
        x = self.transformer.drop(token_embds + positional_embds)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        """
        Crop the block size if necessary
        Args:
            block_size: desired block size after cropping
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                # last 2 dimensions represent length
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Loads a our model parameters from a pre-trained model
        Args:
            model_type: type of pre-trained model to load (as of now only 'gpt2' is an option)
            override_args: arguments to override in the model configuration
        """
        assert model_type in {'gpt2', 'gpt2-xl'} # add other models to the dictionary if required
        override_args = override_args or {}
        assert all(a == 'dropout' for a in override_args)
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained model: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
        }[model_type]
        # update static arguments with values same as openai's gpt2
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        
        # create a new GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        # get the state dictionary, which includes both trainable params and buffers
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith('.attn.bias')] # ignore the mask and buffer

        # initialize a new model using (huggingface's) transformer
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_hf = model_hf.state_dict()
        state_dict_keys_hf = state_dict_hf.keys()
        # copy over tensors from huggingface to our model
        state_dict_keys_hf = [k for k in state_dict_keys_hf if not k.endswith('.attn.masked_bias')] # ignore the buffer
        state_dict_keys_hf = [k for k in state_dict_keys_hf if not k.endswith('.attn.bias')] # ignore the mask
        # openai uses conv layer instead of our linear layer, so we need to transpose the weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(state_dict_keys_hf) == len(state_dict_keys), f"mismatched keys: {len(state_dict_keys_hf)} != {len(state_dict_keys)}"
        for k in state_dict_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose parameters in the hf dictionary and then copy over
                assert state_dict_hf[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k].t())
            else:
                # normal copy over
                assert state_dict_hf[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k])

        return model # model with same parameters as pretrained openai checkpoints

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """
        Set up the optimizer to train the model 
        Args:
            weight_decay: weight decay to be applied to the parameters
            lr: learning rate for the optimizer
            betas: beta coefficient used for computing running averages of the gradient in Adam
            device_type: type of device being used, cpu or gpu
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require gradients
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # any parameters that are 2+ dimensional will be weight decayed, others won't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # Create AdamW optimizer and use the fused version if cuda is available for better performance
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None):
        """ 
        Generate function which takes in a sequence of indeces and predicts the next token in the sequence, max_new_tokens times 
        Args:
            index: input tensor of shape (B, T) containing the initial token indices
            max_new_tokens: maximum number of new tokens to generate
            temperature: scaling factor for logits to control randomness (default is 1.0).
            top_k: if specified, restricts sampling to the top k logits.
        """
        # index.shape = (B, T)
        for _ in range(max_new_tokens):
            # crop index to last block
            if index.size(1) <= self.config.block_size:
                index_cropped = index
            else:
                index_cropped = index[:, -self.config.block_size:]
            # get logit predictions by forwarding the model
            logits, loss = self(index_cropped)
            # reduce to the last time dimension and scale by the temperature
            logits = logits[:, -1, :] / temperature # (B, C)
            # crop the logits if only top k tokens are requested
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # v is the final value
                logits[logits < v[:, [-1]]] = -float('Inf')
            # get probabilities by applying softmax
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the probability distribution
            next_index = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, next_index), dim=1) # (B, T+1)
        return index
