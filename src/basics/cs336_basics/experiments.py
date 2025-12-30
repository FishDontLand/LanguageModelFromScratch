
"""
Main Experiments
"""
import pickle
import threading
import time
import numpy as np
import torch
import os
from tqdm import tqdm
from multiprocessing import Pool

from cs336_basics.optimizer import SGD
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


def trian_bpe_tinystories():
    pass

def train_bpe_expts_owt():
    pass


def tokenizer_experiment_a():
    # Experiment a : compute the compression ratio between tiny story tokenizer and open-web tokenizer
    tiny_story_valid_path = r'../data/TinyStoriesV2-GPT4-valid.txt'
    open_web_text_valid_path = r'../data/owt_valid.txt'
    num_samples = 10
    tiny_stories = get_sample_documents(tiny_story_valid_path, num_samples)
    open_web_text = get_sample_documents(open_web_text_valid_path, num_samples)

    tiny_story_tokenizer = Tokenizer.from_files(r'../data/TinyStoriesV2-GPT4-train_vocab.pkl',
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl',
                                                special_tokens=["<|endoftext|>"])
    open_web_tokenizer = Tokenizer.from_files(r'../data/owt_train_vocab.pkl',
                                              r'../data/owt_train_merges.pkl',
                                              ["<|endoftext|>"])

    tiny_story_encoded = [tiny_story_tokenizer.encode(d) for d in tiny_stories]
    open_web_encoded = [open_web_tokenizer.encode(d) for d in open_web_text]

    tiny_story_compression_ratio = [get_compression_ratio(doc, encoded_doc) for doc, encoded_doc in zip(tiny_stories, tiny_story_encoded)]
    open_web_compression_ratio = [get_compression_ratio(doc, encoded_doc) for doc, encoded_doc in zip(open_web_text, open_web_encoded)]

    print("Tiny Story Compression Ratio")
    print(sum(tiny_story_compression_ratio) / num_samples)

    print("Open-Web Compression Ratio")
    print(sum(open_web_compression_ratio) / num_samples)

def tokenizer_experiment_b():
    # Experiment b: apply tiny story tokenizer on open web text data and compute its compression ratio
    open_web_text_valid_path = r'../data/owt_valid.txt'
    num_samples = 10
    open_web_text = get_sample_documents(open_web_text_valid_path, num_samples)

    tiny_story_tokenizer = Tokenizer.from_files(r'../data/TinyStoriesV2-GPT4-train_vocab.pkl',
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl',
                                                ["<|endoftext|>"])

    open_web_encoded_by_tiny_story = [tiny_story_tokenizer.encode(d) for d in open_web_text]
    open_web_by_tiny_story_compression_ratio = [get_compression_ratio(doc, encoded_doc) for doc, encoded_doc in
                                                zip(open_web_text, open_web_encoded_by_tiny_story)]
    print("Tiny Story Compression Ratio on Open-Web-Text Data")
    print(sum(open_web_by_tiny_story_compression_ratio) / num_samples)

def tokenizer_experiment_c():
    # Experiment c: compute the throughput of the two tokenizers
    tiny_story_tokenizer = Tokenizer.from_files(r'../data/TinyStoriesV2-GPT4-train_vocab.pkl',
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl',
                                                ["<|endoftext|>"])
    open_web_tokenizer = Tokenizer.from_files(r'../data/owt_train_vocab.pkl', r'../data/owt_train_merges.pkl',
                                              ["<|endoftext|>"])

    tiny_story_valid_path = r'../data/TinyStoriesV2-GPT4-valid.txt'
    open_web_text_valid_path = r'../data/owt_valid.txt'
    num_samples = 10
    tiny_stories = get_sample_documents(tiny_story_valid_path, num_samples)
    open_web_text = get_sample_documents(open_web_text_valid_path, num_samples)

    tiny_story_throughput = calc_throughput(tiny_story_tokenizer, tiny_stories)
    open_web_throughput = calc_throughput(open_web_tokenizer, open_web_text)

    print("Tiny Story Tokenizer Throughput: {}".format(tiny_story_throughput))
    print("Open-Web Tokenizer Throughput: {}".format(open_web_throughput))

def tokenizer_experiment_d():
    data_path = r'../data'

    train_data_name_lst = ['TinyStoriesV2-GPT4-train', 'TinyStoriesV2-GPT4-train',
                           'owt_train', 'owt_train']
    data_to_decode_lst = ['TinyStoriesV2-GPT4-train', 'TinyStoriesV2-GPT4-valid',
                          'owt_train', 'owt_valid']

    for train_data_name, data_to_decode in zip(train_data_name_lst, data_to_decode_lst):
        if data_to_decode != 'TinyStoriesV2-GPT4-valid':
            continue
        data_file = r'{}/{}.txt'.format(data_path, data_to_decode)

        tokenizer = Tokenizer.from_files(r'../data/' + f'{train_data_name}_vocab.pkl',
                                         r'../data/' + f'{train_data_name}_merges.pkl',
                                         ['<|endoftext|>'])

        num_chunks = 1000
        output = []
        with open(data_file, 'rb') as f:
            chunks_boundaries = find_chunk_boundaries(f, num_chunks, b'<|endoftext|>')
            num_chunks = len(chunks_boundaries) - 1

            for chunk_idx in tqdm(range(num_chunks)):
                start = chunks_boundaries[chunk_idx]
                end = chunks_boundaries[chunk_idx + 1]
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_encoded = tokenizer.encode(chunk)
                output.append(np.array(chunk_encoded, dtype='uint16'))

        np.save(f'../data/{data_to_decode}_tokenized.npy', np.concatenate(output))


def learning_rate_tuning(lr):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    for t in range(100):
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

"""
Helper Functions
"""
def get_sample_documents(doc_path: str, num_samples: int) -> list[str]:
    with open(doc_path, 'r') as f:
        content = f.read()
        documents = content.split('<|endoftext|>')[:num_samples]
    return documents

def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding='utf-8'))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def calc_throughput(tokenizer: Tokenizer, documents: list[str]) -> float:
    throughputs = []
    for doc in documents:
        start = time.time()
        tokenizer.encode(doc)
        end = time.time()
        run_time = end - start
        throughputs.append(len(doc) / run_time)
    return sum(throughputs) / len(throughputs)

def encode_tokens(chunk, vocab_file, merge_file, job_idx):
    tokenizer = Tokenizer.from_files(vocab_file, merge_file, ["<|endoftext|>"])
    chunk_encoded = tokenizer.encode(chunk)
    return job_idx, np.array(chunk_encoded, dtype='uint16')


if __name__ == '__main__':
    tokenizer_experiment_d()
