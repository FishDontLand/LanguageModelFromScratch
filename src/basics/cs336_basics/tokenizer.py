import multiprocessing
import pickle
from typing import Iterable, Iterator

import regex as re

from cs336_basics.bpe_training import pre_tokenizaiton_core
from cs336_basics.utils import PAT, create_linked_lst


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
        special_tokens = None if special_tokens is None else sorted(special_tokens, key=lambda x : len(x), reverse=True)
        self.special_token_pattern = None if special_tokens is None or len(special_tokens) == 0 else '|'.join([re.escape(tk) for tk in special_tokens])

    @classmethod
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
        output = []
        """
        an implementation without needing to store the split text
        """
        # if self.special_token_pattern is not None:
        #     special_token_pattern = re.compile(self.special_token_pattern).pattern
        #     pre_token_pattern = re.compile(PAT).pattern
        #     end_pattern = f'({pre_token_pattern})({special_token_pattern})'
        #     full_pattern =  f'({end_pattern})|({special_token_pattern})|({pre_token_pattern})'
        #     for match in re.finditer(full_pattern, text):
        #         if re.fullmatch(special_token_pattern, match.group(0)) is not None:
        #             output.append(self.rev_vocab_map[match.group(0).encode('utf-8')])
        #         elif re.fullmatch(pre_token_pattern, match.group(0)) is not None:
        #             encoded_token = self.token_encode(match.group(0))
        #             for n in encoded_token:
        #                 output.append(n)
        #         else:
        #             match = re.fullmatch(end_pattern, match.group(0))
        #             encoded_token = self.token_encode(match.group(1))
        #             for n in encoded_token:
        #                 output.append(n)
        #             output.append(self.rev_vocab_map[match.group(2).encode('utf-8')])
        #
        # else:
        #     for match in re.finditer(PAT, text):
        #         encoded_token = self.token_encode(match.group(0))
        #         for n in encoded_token:
        #             output.append(n)
        #
        # return output
        """
        naive implementation of encoding step
        """
        chunk_start = 0
        if self.special_token_pattern is not None:
            for special_token_match in re.finditer(self.special_token_pattern, text):
                special_token_span = special_token_match.span()
                chunk_end = special_token_span[0]
                if chunk_start < chunk_end:
                    chunk = text[chunk_start:chunk_end]
                    for pre_token_match in re.finditer(PAT, chunk):
                        encoded_text = self.token_encode(pre_token_match.group(0))
                        for n in encoded_text:
                            output.append(n)

                output.append(self.rev_vocab_map[special_token_match.group(0).encode('utf-8')])
                chunk_start = special_token_span[1]

        if chunk_start < len(text):
            chunk = text[chunk_start:]
            for pre_token_match in re.finditer(PAT, chunk):
                encoded_match = self.token_encode(pre_token_match.group(0))
                for n in encoded_match:
                    output.append(n)
        return output


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for element in iterable:
            result = self.encode(element)
            for token_id in result:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        output_bytes = b''.join([self.vocab[vocab_id] for vocab_id in ids])
        return output_bytes.decode('utf-8', errors='replace')

