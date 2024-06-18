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

## Model:
The model defined in `model.py` closely follows the transformer model defined in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper. However, the encoder part has been omitted due to lack of resources - making this is a decoder-only transformer. This makes the model more of a "text completer" instead of a "question answerer". Making the model a "question answerer" requires a fine-tuning stage with (according to my understanding of OpenAI's process) a huge manual overhead.

## GPU and CUDA Optimizations:
**Distributed data parallel:** 

**Mixed precision:** The model uses both 16-bit and 32-bit floating points to reduce memory usage and accelerate training, as defined in NVIDIA's GPU guides. The mantissa of PyTorch tensor weights is truncated to drop the precision but still maintaining high quality.

**Kernel fusion:** Python's interpreter introduces inefficiencies when performing arithmetic of PyTorch tensors, wherein it launches multiple kernels to evaluate every operation in a matrix/equation. Thus, `torch.compile` and flash attention are used to fuse these kernels - reducing computational overhead.

**Smart numbers:** Numbers are intentionally chosen to have as many powers of two as possible since most GPUs are more efficient at processing in 2^n batches. This is the reason for the model's overall token count differing from OpenAI's implementation (dummy tokens were added to increase efficiency.)


## References:
1. GPT-2 paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. GPT-3 paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
4. HuggingFace's GPT-2 Implementation: [huggingface-transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
5. OpenAI's TensorFlow GPT-2 Implementation: [gpt-2](https://github.com/openai/gpt-2/blob/master/src/model.py)

**Special** reference to the man, the myth, the legend [Andrej Kaparthy](https://karpathy.ai/). His [online lectures](https://karpathy.ai/zero-to-hero.html) were my primary source of learnings related to neural networks, deep learning and language models. This project would have been impossible without his (somehow) free lectures.

Note: the assert statements you might see in the code are a part of a coding technique called negative space programming. It programs invariance into code to guarantee a specific behavior.  If the code errors out, its much easier to track down the bug. Making these assertions about required states governed by logic have been very helpful to me when writing large scale ML code like this.