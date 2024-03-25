import torch

# initialization of vocabulary, encoder and decoder
with open('input.txt', 'r', encoding = 'utf-8') as f:
    input = f.read()

vocab = sorted(list(set(input)))
vocab_size = len(vocab)

# create char -> int encoder and int -> char decoder
char_to_int = {ch:i for i,ch in enumerate(vocab)}
int_to_char = {i:ch for i,ch in enumerate(vocab)}
encode = lambda str: [char_to_int[c] for c in str] # encodes a string into a list of integers
decode = lambda ls: ''.join([int_to_char[i] for i in ls]) # decodes a list of integers into a string

# split train and test data (90% split)
data = torch.tensor(encode(input), dtype=torch.long)
n = int(0.9(len(data)))
train_data = data[:n]
validation_data = data[:n]
