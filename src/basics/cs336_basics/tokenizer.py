import multiprocessing
import pickle
from typing import Iterable, Iterator

import regex as re

from cs336_basics.tokenizer_utils import create_linked_lst, PAT


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None=None):
        """
        Constructor of BPE Tokenizer
        Args:
            vocab: vocabulary dictionary
            merges:
            special_tokens:
        """
        self.vocab = vocab
        self.rev_vocab_map = {v: k for k, v in vocab.items()}
        self.merges = {k: v for v, k in enumerate(merges)}
        self.special_tokens = special_tokens
        self.special_token_pattern = '|'.join(self.special_tokens)

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]|None=None):
        vocab = pickle.load(open(vocab_filepath, "rb"))
        merges = pickle.load(open(merges_filepath, "rb"))
        return Tokenizer(vocab, merges, special_tokens=special_tokens)


    def token_encode(self, token: str) -> list[int]:
        head = create_linked_lst(token.encode("utf-8"))

        while True:
            next_merge = None
            next_merge_idx = None
            current = head
            while current is not None and current.next is not None:
                to_merge = (current.val, current.next.val)
                if to_merge in self.merges:
                    if next_merge is None:
                        next_merge = (current.val, current.next.val)
                        next_merge_idx = self.merges[to_merge]
                    elif self.merges[to_merge] < next_merge_idx:
                        next_merge = (current.val, current.next.val)
                        next_merge_idx = self.merges[to_merge]
                current = current.next

            if next_merge is None:
                break
            current = head
            while current is not None and current.next is not None:
                if current.val == next_merge[0] and current.next.val == next_merge[1]:
                    current.val = current.val + current.next.val
                    current.next = current.next.next
                current = current.next

        current = head
        output = []
        while current is not None:
            output.append(self.rev_vocab_map[current.val])
            current = current.next

        return output


    def encode(self, text: str) -> list[int]:
        chunk_lst = re.split(self.special_token_pattern, text)
        output = []
        for chunk in chunk_lst:
            for match in re.finditer(PAT, chunk):
                encoded_match = self.token_encode(match.group(0))
                output += encoded_match
        return output


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for element in iterable:
            result = self.encode(element)
            for token_id in result:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        output_bytes = b''.join([self.vocab[vocab_id] for vocab_id in ids])
        return output_bytes.decode('utf-8', errors='replace')

