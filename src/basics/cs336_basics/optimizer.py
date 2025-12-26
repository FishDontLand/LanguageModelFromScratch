import math
from typing import Optional, Callable

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
             raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state['t'] = t+ 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple, weight_decay: float, eps: float=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        beta1, beta2 = betas
        if beta1 < 0 or beta1 > 1:
            raise ValueError(f"Invalid learning rate: {beta1}")

        if beta2 < 0 or beta2 > 1:
            raise ValueError(f"Invalid learning rate: {beta2}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0:
            raise ValueError(f"Invalid lambda value: {weight_decay}")

        defaults = {"alpha": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "lamb": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                state['m'] = group['beta1'] * state['m'] + (1 - group['beta1']) * p.grad

                if 'v' not in state:
                    state['v'] = torch.zeros_like(p, dtype=p.dtype, device=p.device)
                state['v'] = group['beta2'] * state['v'] + (1 - group['beta2']) * (p.grad ** 2)

                if 't' not in state:
                    state['t'] = 1
                else:
                    state['t'] += 1

                adjust_alpha = group['alpha'] * (math.sqrt(1 - group['beta2'] ** state['t'])) / (1 - group['beta1'] ** state['t'])
                p.data -= adjust_alpha * state['m'] / (torch.sqrt(state['v']) + group['eps'])
                p.data -= group['alpha'] * group['lamb'] * p.data

        return loss
