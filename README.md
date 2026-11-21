# Language Modeling from Scratch

This project follows Stanford CS336 and packages the language model, training systems, and reasoning post-training implementations in one Python distribution. [Stanford CS336](https://stanford-cs336.github.io/)

| Package | Contents |
| --- | --- |
| `cs336.lm` | Byte-pair encoding, a decoder-only Transformer, AdamW, checkpointing, and generation |
| `cs336.systems` | Profiling, FlashAttention-2, distributed data parallelism, and optimizer state sharding. Uses the model and optimizer in `cs336.lm`. |
| `cs336.post_training` | Supervised fine-tuning, expert iteration, GRPO, and math-answer grading |

## Install

Use Python 3.11 or 3.12 and run the install from the repository root.

```bash
python -m pip install -e .
```

```python
from cs336.lm.module import TransformerLM
from cs336.systems.ddp import BucketDDP
from cs336.post_training.grpo import compute_grpo_clip_loss
```

Install `.[test]` to run the test suite with `python -m pytest`. Distributed CPU tests require local TCP sockets. Triton tests and GPU benchmarks require compatible NVIDIA hardware. Install `.[benchmarks]` to run the benchmark scripts in `benchmarks/systems/`.

## Post-training

The tensor-level SFT and GRPO helpers can be used without vLLM. To run rollouts, install `.[post-training]` on Linux x86-64 with two compatible NVIDIA GPUs. The runners put the policy on CUDA device 0 and vLLM on device 1.

```bash
python -m pip install -e ".[post-training]"
python -m cs336.post_training.prepare --data-dir data
python -m cs336.post_training.grpo
```

`run_rl` and `run_expert_iteration` accept a `model_id` or local checkpoint path and a `data_dir` containing `train.jsonl` and `test.jsonl`. The model defaults to Qwen2.5-Math-1.5B. The math grader retains its upstream attribution in the source.
