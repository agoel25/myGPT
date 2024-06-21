# myGPT
My implementation of a generative pre-trained transformer (that actually beats OpenAI's GPT-2 125M parameter model in accuracy).

This repository is a one stop shop to understand how ChatGPT works behind the curtains - from model definition `model.py`, to training `training.py`, to CUDA optimization. The model is currently setup to reproduce OpenAI's GPT-2, however the code can easily be configured to train new models or finetune pretrained models on different datasets.

Here is a sample interaction with myGPT:
```
Prompt: What is a neural network?

Response: A neural network (or neural network model) is a statistical model, which, when
manipulated by a computer, performs an action. These models were originally devised in the
1950s by John von Neumann. They were used for complex behavioral tasks such as speech
recognition. Many popular networks and algorithms, such as the one used to identify faces
in images, were derived from these models. (truncated for readability)
```

## Model
The model closely follows the transformer model defined in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. However, the encoder part has been omitted due to lack of resources - making this is a decoder-only transformer. This makes the model more of a "text completer" instead of a "question answerer". Making the model a "question answerer" requires a fine-tuning stage with (according to my understanding of OpenAI's process) a big manual overhead.

## GPU and CUDA Optimizations
**Distributed data parallel:** DDP is a data parallelism technique which helps train models across multiple machines. This is a multiplicative improvement in computation and efficiency. Some environment variables need to be book-kept for process logistics but PyTorch helps with that.

**Mixed precision:** The model uses both 16-bit and 32-bit floating points to reduce memory usage and accelerate training, as defined in NVIDIA's GPU guides. The mantissa of PyTorch tensor weights is truncated to drop the precision while maintaining reasonable quality.

**Kernel fusion:** Python's interpreter introduces inefficiencies when performing arithmetic of PyTorch tensors, wherein it launches multiple kernels to evaluate every operation in a matrix/equation. Thus, `torch.compile` and flash attention are used to fuse these kernels - reducing computational overhead.

**Smart numbers:** Numbers are intentionally chosen to have as many powers of two as possible since most GPUs are more efficient at processing in 2^n batches. This is the reason for the model's overall token count differing from OpenAI's implementation (dummy tokens were added to increase efficiency.)

***Note***: the assert statements you might see in the code are a part of a coding technique called negative space programming. It programs invariance into code to guarantee a specific behavior.  If the code errors out, its much easier to track down the bug. Making these assertions about required states governed by logic have been very helpful to me when writing large scale ML code like this.

## Quick Start
**Install dependencies**
```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

**Tokenize the dataset**

Tokenization breaks down the text into small pieces, these pieces are what the language model tries to contextualize and to predict. I am using the [Tiktoken](https://github.com/openai/tiktoken) tokenizer and the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) dataset as this is the closest reproduction of OpenAI's GPT setup. This creates the train and validation splits of our dataset and stores them in two binary files: `train.bin` and `val.bin`.
```
python data/openwebtext/prepare.py
```

**Train the model**

Training the model heavily depends on the type of hardware the code runs on, every hardware has slightly different commands to initiate training. The configuration parameters can be adjusted depending on the power of processors. To start your training run from a pre-defined baseline config, just point to the appropriate file in the config folder.

If you do not have a GPU, you can use [lambdalabs.com](https://lambdalabs.com/) to train your models on the cloud. 

I have a normal CPU
```
python train.py config/train_gpt2.py --device=cpu --compile=False --block_size=256 --batch_size=4 --n_layer=4 --n_head=4
```

I have an Apple Silicone CPU (brrr)
```
python train.py config/train_gpt2.py --device=mps --compile=False --batch_size=4
```

I have a single-node GPU (brrrrrr)
```
python train.py config/train_gpt2.py --device=cuda
```

I have a cluster environment with multiple GPU nodes (brrrrrrrrrrrr)
```
# MULTIPLE NODES
torchrun --standalone --nproc_per_node=[num_proc] train.py config/train_gpt2.py

# CLUSTER ENVIRONMENT
# Run on the first (master) node:
torchrun --nproc_per_node=[num_proc] --nnodes=[num_nodes] --node_rank=0 --master_addr=[ip_addr] --master_port=[port] train.py
# Run on the worker node:
torchrun --nproc_per_node=[num_proc] --nnodes=[num_nodes] --node_rank=1 --master_addr=[ip_addr] --master_port=[port] train.py
```
This will run for around 5 days on an NVIDIA 8X A100 GPU and go down to a loss of ~2.85. 

**Sample from the model**

The `sample.py` script is capable of either generating samples from OpenAI's GPT-2 models or the model trained by you. Below is an example command to sample from the model. It uses OpenAI's gpt2-xl model to generate text.
```
python sample.py --init_from=gpt2-xl --start=[your_prompt]
```
To sample from the model you trained, simply run
```
python sample.py
```

Lastly, if you are a pytorch beginner, it is advised to refer to `model_trivialised.py` and `train_trivialised.py` instead of the main model and train files. They focus on the main code while omitting a lot of the optimization code. Therefore, they will be much easier to understand. However, they cannot be trained on our OpenWebText dataset since the code is not optimized.

## References
1. GPT-2 paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. GPT-3 paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
4. HuggingFace's GPT-2 Implementation: [huggingface-transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
5. OpenAI's TensorFlow GPT-2 Implementation: [gpt-2](https://github.com/openai/gpt-2/blob/master/src/model.py)
6. **Special** reference to [Andrej Kaparthy](https://karpathy.ai/). His [online lectures](https://karpathy.ai/zero-to-hero.html) were my primary source of learnings related to neural networks, deep learning and language models.