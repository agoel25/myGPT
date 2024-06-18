# myGPT
This is my implementation of a generative pre-trained transformer (that actually beats OpenAI's comparative GPT-2 model in accuracy).

This repository is a one stop shop to understand how ChatGPT works behind the curtains - from model definition (`model.py`), to training (`training.py`), to CUDA optimization. Currently setup to reproduce OpenAI's GPT-2 model, the code can easily be configured to train new models or simply finetune pretrained models on different datasets. 

Here is a sample interaction with myGPT:
```
Prompt: What is a neural network?

Response: A neural network (or neural network model) is a statistical model, which, when
manipulated by a computer, performs an action. These models were originally devised in the
1950s by John von Neumann. They were used for complex behavioral tasks such as speech
recognition. Many popular networks and algorithms, such as the one used to identify faces
in images, were derived from these models. (truncated for readability)
```

Achievements:
1. Validation loss (negative log likelihood or cross entropy) of ~2.85
2. 

Learnings:


Special reference to the man, the myth, the legend Andrej Kaparthy. His online lectures were my primary source of learnings related to neural networks, deep learning and language models. This project would have been impossible without his (somehow) free lectures.

Note: the assert statements you might see in the code are a part of a coding technique called negative space programming. It programs invariance into code to guarantee a specific behavior.  If the code errors out, its much easier to track down the bug. Making assertions about required states governed by calculations or logic have been very helpful to me when writing large scale ML code like this.