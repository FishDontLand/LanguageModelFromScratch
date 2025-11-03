import regex as re
from heapq import heappush, heappop, heapify
from multiprocessing import Process, Manager

from cs336_basics.pretokenization_example import find_chunk_boundaries


class Node:
    def __init__(self):
        self.val = None
        self.next = None

class BPE_Tokenizer:
    def __init__(self, max_vocab_size, special_tokens, num_processes=4):
        self.max_vocab_size = max_vocab_size
        self.special_tokens = set(special_tokens)
        self.special_token_pattern = '|'.join(special_tokens)
        self.pre_token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._initialize_members()
        self.num_processes = num_processes

    def _initialize_members(self):
        self.vocab = dict()
        self.merges = list()

    def get_pre_tokenization_count(self, chunk, output):
        pre_tokenization_count = dict()
        block_start = 0

        for special_token_match in re.finditer(self.special_token_pattern, chunk):
            span = special_token_match.span()
            block_end = span[0]
            str_block = chunk[block_start:block_end]
            for pre_token_match in re.finditer(self.pre_token_pattern, str_block):
                pre_token = pre_token_match.group(0)
                if pre_token not in pre_tokenization_count:
                    pre_tokenization_count[pre_token] = 0
                pre_tokenization_count[pre_token] += 1

            block_start = span[1]

        if block_start < len(chunk):
            block_end = len(chunk)
            str_block = chunk[block_start:block_end]

            for pre_token_match in re.finditer(self.pre_token_pattern, str_block):
                pre_token = pre_token_match.group(0)
                if pre_token not in pre_tokenization_count:
                    pre_tokenization_count[pre_token] = 0
                pre_tokenization_count[pre_token] += 1

        output.append(pre_tokenization_count)

    def fit(self, file, verbose=False):
        self._initialize_members()
        boundaries = find_chunk_boundaries(file, self.num_processes, b'<|endoftext|>')
        if verbose:
            print("boundaries", boundaries)
        manager = Manager()
        batch_pre_token_count_lst = manager.list()
        all_processes = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            all_processes.append(Process(target=self.get_pre_tokenization_count, args=(chunk, batch_pre_token_count_lst)))

        for p in all_processes:
            p.start()
            p.join()

        # for p in all_processes:
        #     p.join()

        pre_tokenization_count = dict()
        for batch_count in batch_pre_token_count_lst:
            for k, v in batch_count.items():
                if k not in pre_tokenization_count:
                    pre_tokenization_count[k] = v
                else:
                    pre_tokenization_count[k] += v

        if verbose:
            print("pre_tokenization_count", pre_tokenization_count)
        merge_count = dict()
        pre_token_linked_lst = dict()
        for word, count in pre_tokenization_count.items():
            word_bytes = word.encode('utf-8')
            head = None
            current = None
            for i in range(len(word_bytes)):
                if i < len(word_bytes) - 1:
                    merge = tuple([word_bytes[i:i + 1], word_bytes[i + 1:i+2]])
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
        for token in self.special_tokens:
            self.vocab[next_id] = token.encode("utf-8")
            next_id += 1

        for i in range(256):
            self.vocab[next_id] = bytes([i])
            next_id += 1

        while num_merges < self.max_vocab_size - len(self.special_tokens) - 256:
            if verbose:
                print("merge count: ", merge_count)
            best_merge = None
            max_count = -1
            for k, v in merge_count.items():
                if best_merge is None or v > max_count:
                    best_merge = k
                    max_count = v
                elif v == max_count and k[0] + k[1] > best_merge[0] + best_merge[1]:
                    best_merge = k
                    max_count = v

            if best_merge is None:
                raise Exception("no merge found!")
            self.merges.append(best_merge)
            merged_token = best_merge[0] + best_merge[1]
            self.vocab[next_id] = merged_token
            num_merges += 1
            next_id += 1

            if verbose:
                print("best merge: ", best_merge[0], best_merge[1])
                print("merged token: ", merged_token)
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



if __name__ == '__main__':
    import io
    example_text = """
    low low low low low
    lower lower widest <|endoftext|> widest widest
    newest newest newest newest newest newest <|endoftext|>"""
    example_text_binary = example_text.encode("utf-8")
    example_io_object = io.BytesIO(example_text_binary)
    special_token_lst = [" ", "\n", r"<\|endoftext\|>"]
    num_merges = 6
    bpe_tokenizer = BPE_Tokenizer(len(special_token_lst) + 256 + num_merges, special_token_lst, 2)
    bpe_tokenizer.fit(example_io_object, False)
    for k, v in bpe_tokenizer.vocab.items():
        if k >= len(bpe_tokenizer.special_tokens) + 256:
            print(v.decode("utf-8"))