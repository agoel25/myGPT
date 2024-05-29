import os
from tqdm import tqdm # for progress bar
import numpy as np
import tiktoken # openai's tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers for parallel processing
num_workers = 6

# use tiktoken to get the same tokenization encoding as OpenAI's gpt2
encoding = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # load the openwebtext dataset
    dataset = load_dataset("openwebtext", num_proc=num_workers)

    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # function to get encoding ids after tokenization
    def process(example):
        ids = encoding.encode_ordinary(example['text']) # ignores special tokens
        ids.append(encoding.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    # tokenize the data
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing data",
        num_proc=num_workers,
    )

    # write ids into data files for train and val splits
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        # create a memory mapped array for efficient file writing
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,)) 
        # splitting dataset into managable batches for processing
        total_batches = 1024

        # write data (ids) in batches
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            # batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # write into the memory mapped array
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
