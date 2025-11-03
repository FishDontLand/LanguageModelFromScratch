from multiprocessing.managers import ListProxy
from typing import Tuple
import time
import regex as re
from multiprocessing import Process, Manager
from cs336_basics.pretokenization_example import find_chunk_boundaries

class Node:
    """
    helper class for linked list
    """
    def __init__(self):
        self.val = None
        self.next = None

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_pre_tokenization_count(split_pattern: str, pre_token_pattern: str, chunk: str, output: ListProxy):
    """
    function that returns the number of tokens in chunk

    Args:
        split_pattern: pattern used to split the text
        pre_token_pattern: pattern used to identify pre tokens
        chunk: chunk of text to be tokenized
        output:

    Returns:

    """
    pre_tokenization_count = dict()

    splits = re.split(split_pattern, chunk)
    for e in splits:
        for pre_token_match in re.finditer(pre_token_pattern, e):
                pre_token = pre_token_match.group(0)
                if pre_token not in pre_tokenization_count:
                    pre_tokenization_count[pre_token] = 0
                pre_tokenization_count[pre_token] += 1

    output.append(pre_tokenization_count)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    train BPE tokenizer

    Args:
        input_path: input path to the corpus file
        vocab_size: maximum vocabulary size
        special_tokens: list of special tokens. only used for splitting, but not bpe training.

    Returns:
        vocab: The tokenizer vocabulary, a mapping from int to bytes.
        merges: A list of BPE merges produced from training.

    """
    vocab = dict()
    merges = list()
    num_processes = 4

    point1 = time.time()

    manager = Manager()
    batch_pre_token_count_lst = manager.list()
    all_processes = []
    split_pattern = '|'.join(special_tokens)
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b'<|endoftext|>')

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore").replace('\r\n', '\n')
            all_processes.append(Process(target=get_pre_tokenization_count, args=(split_pattern, PAT, chunk, batch_pre_token_count_lst)))

        for p in all_processes:
            p.start()

        for p in all_processes:
            p.join()

    point2 = time.time()
    print('pretokenization: ', point2 - point1)

    pre_tokenization_count = dict()
    for batch_count in batch_pre_token_count_lst:
        for k, v in batch_count.items():
            if k not in pre_tokenization_count:
                pre_tokenization_count[k] = v
            else:
                pre_tokenization_count[k] += v

    merge_count = dict()
    pre_token_linked_lst = dict()
    for word, count in pre_tokenization_count.items():
        word_bytes = word.encode('utf-8')
        head = None
        current = None
        for i in range(len(word_bytes)):
            if i < len(word_bytes) - 1:
                merge = tuple([word_bytes[i:i + 1], word_bytes[i + 1:i + 2]])
                if merge not in merge_count:
                    merge_count[merge] = 0
                merge_count[merge] += count

            node = Node()
            node.val = word_bytes[i:(i + 1)]
            if head is None:
                head = node
                current = head
            else:
                current.next = node
                current = node

        pre_token_linked_lst[word] = head

    num_merges = 0
    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    while num_merges < vocab_size - len(special_tokens) - 256:
        best_merge = None
        max_count = -1
        for k, v in merge_count.items():
            if best_merge is None or v > max_count:
                best_merge = k
                max_count = v
            elif v == max_count and k > best_merge:
                best_merge = k
                max_count = v

        if best_merge is None:
            raise Exception("no merge found!")

        merges.append(best_merge)
        merged_token = best_merge[0] + best_merge[1]
        vocab[next_id] = merged_token
        num_merges += 1
        next_id += 1

        for pre_token, head in pre_token_linked_lst.items():
            current = head
            while current is not None and current.next is not None:
                if current.val + current.next.val == merged_token:
                    current.val = merged_token
                    current.next = current.next.next

                current = current.next

        merge_count = dict()
        for pre_token, head in pre_token_linked_lst.items():
            current = head
            while current is not None:
                if current.next is not None:
                    to_merge = tuple([current.val, current.next.val])
                    if to_merge not in merge_count:
                        merge_count[to_merge] = 0
                    merge_count[to_merge] += pre_tokenization_count[pre_token]
                current = current.next

    point3 = time.time()
    print('merge: ', point3 - point2)
    return vocab, merges



if __name__ == '__main__':
    # special_token_lst = [" ", "\n", r"<\|endoftext\|>"]
    # num_merges = 6
    # vocab, merges = train_bpe(r"C:\Users\dongz\Documents\CS336\assignment1-basics\data\test data.txt",
    #                           len(special_token_lst) + 256 + num_merges, special_token_lst)
    # for k, v in vocab.items():
    #     if k >= len(special_token_lst) + 256:
    #         print(v.decode("utf-8"))
    input_path = r"C:\Users\dongz\Documents\CS336\assignment1-basics\tests\fixtures\tinystories_sample_5M.txt"
    with open(input_path, "r", encoding="utf-8") as file:
        data = file.read()
    actual_vocab, actual_merges = train_bpe(input_path=input_path,vocab_size=1000,special_tokens=["<|endoftext|>"])
    import pickle
    input_path = r'C:\Users\dongz\Documents\CS336\assignment1-basics\tests\_snapshots\test_train_bpe_special_tokens.pkl'
    with open(input_path, "rb") as f:
        expected_data = pickle.load(f)