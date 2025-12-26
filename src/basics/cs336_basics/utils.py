import math
import os
import typing

import einx
import torch
import numpy as np

class Node:
    """
    helper class for linked list
    """
    def __init__(self):
        self.val = None
        self.next = None
        self.prev = None


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def create_linked_lst(word_bytes):
    head = None
    current = None
    for i in range(len(word_bytes)):
        node = Node()
        node.val = word_bytes[i:(i + 1)]
        if head is None:
            head = node
            current = head
        else:
            current.next = node
            node.prev = current
            current = node

    return head


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_normalized = x - x.max(dim=i, keepdim=True).values
    exp = torch.exp(x_normalized)
    exp_sum = exp.sum(dim=i, keepdim=True)
    return exp / exp_sum


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = K.size(-1)
    inner_product = einx.dot('... q_len d_k, ... k_len d_k -> ... q_len k_len', Q, K) / math.sqrt(d_k)
    inner_product[~mask] = -torch.inf
    importance = softmax(inner_product, -1)
    return einx.dot('... q_len k_len, ... k_len d_v -> ... q_len d_v', importance, V)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    normed_logits = logits - logits.max(dim=-1, keepdim=True).values
    exp_sum = torch.exp(normed_logits).sum(dim=-1)
    numerator = torch.gather(normed_logits, dim=-1, index=targets.view(*targets.size(), 1))
    prob = torch.log(exp_sum).view(*exp_sum.size(), 1) - numerator
    return prob.mean()


def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_warm: int, T_cos: int) -> float:
    if alpha_max < alpha_min:
        raise ValueError(f"alpha_max should be greater or equal to alpha_min")

    if T_cos < T_warm:
        raise ValueError(f"T_cos should be no less than T_warm")

    if t < T_warm:
        return t / T_warm * alpha_max

    if t > T_cos:
        return alpha_min

    return alpha_min + 0.5 * (1 + math.cos((t - T_warm) * math.pi / (T_cos - T_warm))) * (alpha_max - alpha_min)


def gradient_clipping(param_lst: list[torch.Tensor], max_l2: float, eps: float = 1e-6):
    total_norm_square = 0
    for p in param_lst:
        if p.grad is None:
            continue
        p_norm = p.grad.norm()
        total_norm_square += (p_norm * p_norm).detach()

    total_norm = torch.sqrt(total_norm_square)
    if total_norm > max_l2:
        scaler = max_l2 / (total_norm + eps)
        for p in param_lst:
            if p.grad is None:
                continue
            p.grad *= scaler

def data_loading(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size > len(x) - context_length:
        raise ValueError(f"batch_size should be less than {len(x) - context_length}")
    rng = np.random.default_rng()
    samples_starts = rng.integers(0, len(x) - context_length, size=(batch_size , 1))
    input_locs = samples_starts + np.arange(context_length)
    target_locs = input_locs + 1
    inputs = torch.tensor(x[input_locs], device=device, dtype=torch.long)
    targets = torch.tensor(x[target_locs], device=device, dtype=torch.long)
    return inputs, targets



def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    states = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(states, out)

def load_checkpoint(src: str|os.PathLike|typing.BinaryIO|typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    states = torch.load(src)
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['optimizer'])
    return states['iteration']







