import os
import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from heapq import heappush, heappop, heapify

class Node:
    """
    helper class for linked list
    """
    def __init__(self):
        self.val = None
        self.next = None
        self.prev = None

class CustomHeap:
    def __init__(self, heap=[]):
        self.heap = heap
        heapify(self.heap)
        self.cache = set()

    def append(self, node):
        heappush(self.heap, node)

    def lazy_delete(self):
        while self.heap[0] in self.cache:
            e = heappop(self.heap)
            self.cache.remove(e)

    def pop(self):
        self.lazy_delete()
        return heappop(self.heap)

    def delete(self, node):
        self.cache.add(node)


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_batch_pre_tokenization_count(split_pattern, chunk: str):
    splits = re.split(split_pattern, chunk)
    pre_tokenization_count = dict()
    for e in splits:
        for pre_token_match in re.finditer(PAT, e):
            pre_token = pre_token_match.group(0)
            if pre_token not in pre_tokenization_count:
                pre_tokenization_count[pre_token] = 0
            pre_tokenization_count[pre_token] += 1

    return pre_tokenization_count

def pre_tokenizaiton_core(split_pattern, input_path):
    batch_pre_token_count_lst = list()
    num_processes = os.cpu_count()
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(file, num_processes, b'<|endoftext|>')
        pool = Pool(len(boundaries) - 1)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore").replace('\r\n', '\n')
            pool.apply_async(get_batch_pre_tokenization_count,
                             args=(split_pattern, chunk), callback=lambda r: batch_pre_token_count_lst.append(r))

    pool.close()
    pool.join()

    return batch_pre_token_count_lst

def find_best_merge(merge_count):
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

    return best_merge, max_count

def update_status(best_merge, pre_token_linked_lst, merge_count,  word_bytes_map, pre_tokenization_count):
    merged_token = best_merge[0] + best_merge[1]
    words_to_iter = word_bytes_map[merged_token]
    for pre_token in words_to_iter:
        head = pre_token_linked_lst[pre_token]
        base_occurrence = pre_tokenization_count[pre_token]
        current = head
        while current is not None and current.next is not None:
            merged_token = best_merge[0] + best_merge[1]
            if current.val + current.next.val == merged_token:
                if current.prev is not None:
                    key = tuple([current.prev.val, current.val])
                    merge_count[key] -= base_occurrence

                if current.next.next is not None:
                    key = tuple([current.next.val, current.next.next.val])
                    if key != best_merge:
                        merge_count[key] -= base_occurrence

                merge_count[best_merge] = 0

                current.val = merged_token
                current.next.prev = None
                current.next = current.next.next

                if current.next is not None:
                    current.next.prev = current

                if current.prev is not None:
                    key = tuple([current.prev.val, current.val])
                    merge_count[key] = merge_count.get(key, 0) + base_occurrence

                if current.next is not None:
                    key = tuple([current.val, current.next.val])
                    merge_count[key] = merge_count.get(key, 0) + base_occurrence
            current = current.next


def bytes_merge_core(batch_pre_token_count_lst, special_tokens, vocab_size):
    vocab = dict()
    merges = list()
    word_bytes_map = dict()
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
        # word_bytes_map[word] = set([word_bytes[i:j] for i in range(len(word_bytes)) for j in range(i + 1, len(word_bytes) + 1)])
        # word_bytes_map[word] = word_bytes
        for i in range(len(word_bytes)):
            for j in range(i + 1, len(word_bytes) + 1):
                sub_word = word_bytes[i:j]
                if sub_word not in word_bytes_map:
                    word_bytes_map[sub_word] = []
                word_bytes_map[sub_word].append(word)

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
                node.prev = current
                current = node

        pre_token_linked_lst[word] = head

    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    for _ in range(vocab_size - len(special_tokens) - 256):
        best_merge, max_count = find_best_merge(merge_count)

        merges.append(best_merge)
        merged_token = best_merge[0] + best_merge[1]
        vocab[next_id] = merged_token
        next_id += 1

        update_status(best_merge, pre_token_linked_lst, merge_count, word_bytes_map, pre_tokenization_count)

    return vocab, merges

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

    split_pattern = '|'.join(special_tokens)
    batch_pre_token_count_lst = pre_tokenizaiton_core(split_pattern, input_path)
    vocab, merges = bytes_merge_core(batch_pre_token_count_lst, special_tokens, vocab_size)

    return vocab, merges



if __name__ == '__main__':
    # special_token_lst = [" ", "\n", r"<\|endoftext\|>"]
    #num_merges = 6
    # vocab, merges = train_bpe(r"C:\Users\dongz\Documents\CS336\assignment1-basics\data\test data.txt",
    #                           len(special_token_lst) + 256 + num_merges, special_token_lst)
    # for k, v in vocab.items():
    #     if k >= len(special_token_lst) + 256:
    #         print(v.decode("utf-8"))
    # file = r'C:\Users\dongz\Documents\CS336\assignment1-basics\data\test data.txt'
    file = r'C:\Users\dongz\Documents\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt'
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = train_bpe(file, 10000, [r"<\|endoftext\|>"])
    profiler.disable()
    profiler.print_stats()