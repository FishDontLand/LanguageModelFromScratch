import cProfile
import datetime
import os
import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from heapq import heappush, heappop, heapify
import pickle

from cs336_basics.tokenizer_utils import PAT, Node, create_linked_lst


class MergeCountMap:
    """
    helper class designed to store the number of occurrence of merge pairs
    """
    class MergeCount:
        def __init__(self, bytes_pair: tuple[bytes, bytes], count: int):
            self.bytes_pair = bytes_pair
            self.count = count

        def __lt__(self, other):
            return self.get_tuple_rep() > other.get_tuple_rep()

        def get_tuple_rep(self):
            return self.count, self.bytes_pair[0], self.bytes_pair[1]

    def __init__(self, merge_count_map: dict[tuple[bytes, bytes], int]):
        self.merge_count_map = merge_count_map
        self.key_heap = [MergeCountMap.MergeCount(k, v) for k, v in self.merge_count_map.items()]
        heapify(self.key_heap)
        self.cache = set()

    def get(self, pair: tuple[bytes, bytes], default=None):
        return self.merge_count_map.get(pair, default)

    def update(self, pair: tuple[bytes, bytes], count):
        self.merge_count_map[pair] = count
        self.cache.add(pair)

    def pop(self) -> tuple[bytes, bytes] | None:
        for pair in self.cache:
            heappush(self.key_heap, MergeCountMap.MergeCount(pair, self.merge_count_map[pair]))

        self.cache.clear()
        pop = heappop(self.key_heap)
        while pop.count != self.merge_count_map[pop.bytes_pair]:
            if self.key_heap:
                pop = heappop(self.key_heap)
            else:
                pop = None
                break

        return pop.bytes_pair

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]|None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    special_tokens = None if special_tokens is None else sorted(special_tokens, key=lambda x: len(x), reverse=True)
    batch_pre_token_count_lst = pre_tokenizaiton_core(special_tokens, input_path)
    print(datetime.datetime.now(), 'batch tokenization finished')
    return bytes_merge_core(batch_pre_token_count_lst, special_tokens, vocab_size)


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

def pre_tokenizaiton_core(special_tokens, input_path):
    slit_pattern = '|'.join([re.escape(tk) for tk in special_tokens])
    batch_pre_token_count_lst = list()
    num_processes = min(os.cpu_count(), 10)
    with open(input_path, "rb") as file:
        if '<|endoftext|>' in special_tokens:
            boundaries = find_chunk_boundaries(file, num_processes, b'<|endoftext|>')
        else:
            boundaries = find_chunk_boundaries(file, num_processes, special_tokens[0].encode('utf-8'))
        pool = Pool(len(boundaries) - 1)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            pool.apply_async(get_batch_pre_tokenization_count,
                             args=(slit_pattern, chunk), callback=lambda r: batch_pre_token_count_lst.append(r))

    pool.close()
    pool.join()

    return batch_pre_token_count_lst


def update_status(best_merge, merged_token, pre_token_linked_lst, merge_count,  word_bytes_map, search_map, pre_tokenization_count):
    search_map[merged_token] = set()
    # words_to_iter = word_bytes_map[merged_token]
    candidates = search_map[best_merge[0]].intersection(search_map[best_merge[1]])
    for pre_token in candidates:
        head = pre_token_linked_lst[pre_token]
        base_occurrence = pre_tokenization_count[pre_token]
        current = head
        while current is not None and current.next is not None:
            if current.val == best_merge[0] and current.next.val == best_merge[1]:
                search_map[merged_token].add(pre_token)
                if current.prev is not None:
                    key = tuple([current.prev.val, current.val])
                    merge_count.update(key, merge_count.get(key, 0) - base_occurrence)

                if current.next.next is not None:
                    key = tuple([current.next.val, current.next.next.val])
                    if key != best_merge:
                        merge_count.update(key, merge_count.get(key, 0) - base_occurrence)

                merge_count.update(best_merge, 0)

                current.val = merged_token
                current.next.prev = None
                current.next = current.next.next

                if current.next is not None:
                    current.next.prev = current

                if current.prev is not None:
                    key = tuple([current.prev.val, current.val])
                    merge_count.update(key, merge_count.get(key, 0) + base_occurrence)

                if current.next is not None:
                    key = tuple([current.val, current.next.val])
                    merge_count.update(key, merge_count.get(key, 0) + base_occurrence)

            current = current.next


def bytes_merge_core(batch_pre_token_count_lst, special_tokens, vocab_size):
    vocab = dict()
    merges = list()
    word_bytes_map = dict()
    search_map = dict()
    pre_tokenization_count = dict()
    for batch_count in batch_pre_token_count_lst:
        for k, v in batch_count.items():
            if k not in pre_tokenization_count:
                pre_tokenization_count[k] = v
            else:
                pre_tokenization_count[k] += v

    merge_count_dict = dict()
    pre_token_linked_lst = dict()
    for word, count in pre_tokenization_count.items():
        word_bytes = word.encode('utf-8')
        # word_bytes_map[word] = set([word_bytes[i:j] for i in range(len(word_bytes)) for j in range(i + 1, len(word_bytes) + 1)])
        word_bytes_map[word] = word_bytes
        for i in range(len(word_bytes)):
            sub_word = word_bytes[i:i+1]
            if sub_word not in search_map:
                search_map[sub_word] = set()
            search_map[sub_word].add(word)

        for i in range(len(word_bytes)):
            if i < len(word_bytes) - 1:
                merge = tuple([word_bytes[i:i + 1], word_bytes[i + 1:i + 2]])
                if merge not in merge_count_dict:
                    merge_count_dict[merge] = 0
                merge_count_dict[merge] += count

        pre_token_linked_lst[word] = create_linked_lst(word_bytes)
    print(datetime.datetime.now(), 'pretoknization merge finished')
    merge_count_map = MergeCountMap(merge_count_dict)
    next_id = 0

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    for rounds in range(vocab_size - len(special_tokens) - 256):
        verbose = rounds % 1000 == 0
        if verbose:
            print(datetime.datetime.now(), 'num merges: ', rounds + 1, ' finished')
        best_merge = merge_count_map.pop()
        merges.append(best_merge)
        merged_token = best_merge[0] + best_merge[1]
        vocab[next_id] = merged_token
        next_id += 1
        update_status(best_merge, merged_token, pre_token_linked_lst, merge_count_map, word_bytes_map, search_map, pre_tokenization_count)

    return vocab, merges


if __name__ == '__main__':
    # data_name = 'owt_train'
    data_name = 'TinyStoriesV2-GPT4-train'
    data_path = r'C:\Users\dongz\Documents\CS336\assignment1-basics\data'

    data_file = r'{}\{}.txt'.format(data_path, data_name)
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    print(datetime.datetime.now(), 'program started')
    vocab, merges = train_bpe(data_file, 10000, ["<|endoftext|>"])
    profiler.disable()
    profiler.print_stats()

    output_file = r'{}\{}.pkl'.format(data_path, data_name)
    with open(output_file, 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f)
