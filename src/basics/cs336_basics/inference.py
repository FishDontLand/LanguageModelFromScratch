import torch

from cs336_basics.module import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.utils import softmax


@torch.no_grad()
def decode(prompt: torch.Tensor, model: TransformerLM, max_tokens: int, temperature: float,  p: float, tokenizer: Tokenizer):
    output = []
    for _ in range(max_tokens):
        logits = model.forward(prompt)
        prob = softmax(logits / temperature, i=-1)
        prob = prob.select(dim=-2, index=-1)
        prob_sorted, idx = torch.sort(prob, descending=True, dim=-1)
        cum_prob = torch.cumsum(prob_sorted, dim=-1)
        prob_sorted[cum_prob > p] = 0.0
        sample_idx = torch.multinomial(prob_sorted, num_samples=1)
        next_word_idx = idx.gather(dim=-1, index=sample_idx).detach().cpu().item()
        output.append(next_word_idx)
        if next_word_idx == tokenizer.rev_vocab_map['<|endoftext|>'.encode('utf-8')]:
            break
    return torch.tensor(output, dtype=torch.long, device=prompt.device)