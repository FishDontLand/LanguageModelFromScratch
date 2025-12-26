
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
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl')
    open_web_tokenizer = Tokenizer.from_files(r'../data/owt_train_vocab.pkl', r'../data/owt_train_merges.pkl')

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
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl')

    open_web_encoded_by_tiny_story = [tiny_story_tokenizer.encode(d) for d in open_web_text]
    open_web_by_tiny_story_compression_ratio = [get_compression_ratio(doc, encoded_doc) for doc, encoded_doc in
                                                zip(open_web_text, open_web_encoded_by_tiny_story)]
    print("Tiny Story Compression Ratio on Open-Web-Text Data")
    print(sum(open_web_by_tiny_story_compression_ratio) / num_samples)

def tokenizer_experiment_c():
    # Experiment c: compute the throughput of the two tokenizers
    tiny_story_tokenizer = Tokenizer.from_files(r'../data/TinyStoriesV2-GPT4-train_vocab.pkl',
                                                r'../data/TinyStoriesV2-GPT4-train_merges.pkl')
    open_web_tokenizer = Tokenizer.from_files(r'../data/owt_train_vocab.pkl', r'../data/owt_train_merges.pkl')

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
    # Experiment d: encode the training and validationdata  and serialize the token IDs as numpy array for future usage
    tiny_story_train_path = r'../data/TinyStoriesV2-GPT4-train.txt'
    open_web_text_train_path = r'../data/owt_train.txt'
    tiny_story_valid_path = r'../data/TinyStoriesV2-GPT4-valid.txt'
    open_web_text_valid_path = r'../data/owt_valid.txt'

    tiny_story_vocab_file = r'../data/TinyStoriesV2-GPT4-train_vocab.pkl'
    tiny_story_merge_file = r'../data/TinyStoriesV2-GPT4-train_merges.pkl'
    open_web_vocab_file = r'../data/owt_train_vocab.pkl'
    open_web_merge_file = r'../data/owt_train_merges.pkl'

    tiny_story_tokenizer = Tokenizer.from_files(tiny_story_vocab_file, tiny_story_merge_file)
    open_web_tokenizer = Tokenizer.from_files(open_web_vocab_file, open_web_merge_file)
    file_paths = [tiny_story_train_path, tiny_story_valid_path, open_web_text_train_path, open_web_text_valid_path]
    output_file_names = ['TinyStoriesV2-GPT4-train', 'TinyStoriesV2-GPT4-valid', 'owt_train', 'owt_valid']
    vocab_file_lst = [tiny_story_vocab_file, tiny_story_vocab_file, open_web_vocab_file, open_web_vocab_file]
    merge_file_lst = [tiny_story_merge_file, tiny_story_merge_file, open_web_merge_file, open_web_merge_file]
    # tokenizer_lst = [tiny_story_tokenizer, tiny_story_tokenizer, open_web_tokenizer, open_web_tokenizer]



    for path, output_name, vocab_file, merge_file in zip(file_paths, output_file_names, vocab_file_lst, merge_file_lst):
        if output_name != 'TinyStoriesV2-GPT4-train':
            continue
        print("working on generating {}".format(output_name))
        pool = Pool(5)
        file_loc = r'../data/{}_tokenized.npy'.format(output_name)
        num_chunks = 50
        num_processes = 6

        sem = threading.Semaphore(num_processes)

        with open(path, 'rb') as f:
            chunks_boundaries = find_chunk_boundaries(f, num_chunks * num_processes, b'<|endoftext|>')
            num_chunks = len(chunks_boundaries) - 1
            token_seq_lst = [None] * num_chunks
            def callback(res):
                token_seq_lst[res[0]] = res[1]
                sem.release()

            for chunk_idx in tqdm(range(num_chunks)):
                start = chunks_boundaries[chunk_idx]
                end = chunks_boundaries[chunk_idx + 1]
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                sem.acquire()
                pool.apply_async(encode_tokens,
                                 args=(chunk, vocab_file, merge_file, chunk_idx),
                                 callback=callback,
                                 error_callback=lambda e : f'following error occurred during execution of chunk {chunk_idx}: {e}')

            pool.close()
            pool.join()
        full_token_seq = np.concatenate(token_seq_lst)
        np.save(file_loc, full_token_seq)

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
    tokenizer = Tokenizer.from_files(vocab_file, merge_file)
    chunk_encoded = tokenizer.encode(chunk)
    return job_idx, np.array(chunk_encoded, dtype='uint16')


if __name__ == '__main__':
    raw_file = r'../data/TinyStoriesV2-GPT4-train.txt'
    tokenized_file = r'../data/TinyStoriesV2-GPT4-train_tokenized.npy'

    vocab_file = r'../data/TinyStoriesV2-GPT4-train_vocab.pkl'
    merge_file = r'../data/TinyStoriesV2-GPT4-train_merges.pkl'
    tokenizer = Tokenizer.from_files(vocab_file, merge_file)
    with open(raw_file, 'r') as f:
        raw_data = f.read()

    tokenized_data = np.load(tokenized_file)
    print("max id: ", tokenized_data.max())
    print('min id:', tokenized_data.min())

    decoded_text = tokenizer.decode(tokenized_data)
    assert raw_data == decoded_text