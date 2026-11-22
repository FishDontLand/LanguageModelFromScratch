# lmstack

I built a decoder-only language model from scratch in PyTorch, implemented GPU kernels and distributed training primitives, and built a reasoning post-training loop for a pretrained math model. The repository brings together the model, systems code, and experiments for those three parts of the LLM lifecycle.

| Component | What I built |
| --- | --- |
| [Language model from scratch](#language-model-from-scratch) | Byte-level BPE, a decoder-only Transformer, and the pretraining and generation pipeline |
| [Training systems](#training-systems) | FlashAttention-2 in Triton, two implementations of distributed data parallelism, a sharded optimizer, and profiling tools |
| [Reasoning post-training](#reasoning-post-training) | A two-GPU rollout and training loop with vLLM, outcome-based rewards, and clipped GRPO |

## Language model from scratch

- **Tokenizer.** Parallelized pre-token counting across corpus chunks and used a heap, linked lists, and an index of affected words to update pair counts incrementally during byte-pair merges. The tokenizer supports special tokens, UTF-8 round trips, and streaming encoding. [BPE trainer](src/lmstack/lm/bpe_training.py) · [Tokenizer](src/lmstack/lm/tokenizer.py)
- **Transformer.** Implemented embeddings, causal multi-head attention, rotary position embeddings, RMSNorm, and SwiGLU in PyTorch. Stacked pre-normalized residual blocks with a vocabulary projection for next-token prediction. [Model](src/lmstack/lm/module.py)
- **Pretraining and generation.** Implemented cross-entropy, AdamW, gradient clipping, and a warmup-plus-cosine learning-rate schedule. The trainer samples next-token batches from memory-mapped token arrays, logs validation loss and gradient norms, and checkpoints model and optimizer state. The generation code loads a checkpoint and samples with a temperature and cumulative-probability cutoff. [Training](src/lmstack/lm/training.py) · [Optimizer](src/lmstack/lm/optimizer.py) · [Generation](src/lmstack/lm/inference.py)

I wrote experiments on TinyStories and OpenWebText to compare tokenizer compression and encoding throughput, including how the TinyStories tokenizer behaves on OpenWebText. [Tokenizer experiments](src/lmstack/lm/experiments.py)

[BPE optimization commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/91295db8b218889736924bb54dacabe8648952e0) · [Transformer and pretraining commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/aca3a0f43628c68f41e82f4d24e53c2e0d07987b)

## Training systems

- **FlashAttention-2.** Implemented a tiled PyTorch reference and Triton forward and backward kernels. The forward kernel keeps a running maximum and normalization term for online softmax. The backward kernel reconstructs attention probabilities from the saved log-sum-exp and accumulates query, key, and value gradients. [Kernels](src/lmstack/systems/flash_attention.py) · [Reference comparisons](tests/systems/test_attention.py)
- **Distributed training.** Implemented both per-parameter and bucketed gradient synchronization. Autograd hooks launch asynchronous reductions as gradients become ready. The sharded optimizer uses reduce-scatter to distribute gradients, updates each rank’s parameter shard with local optimizer state, and all-gathers the updated parameters. [DDP and optimizer](src/lmstack/systems/ddp.py) · [Distributed tests](tests/systems)
- **Profiling and benchmarks.** Instrumented forward, backward, and optimizer steps with NVTX ranges for Nsight Systems and recorded CUDA memory snapshots. The forward attention benchmark compares the Triton kernel with eager PyTorch across sequence lengths from 128 to 65,536, attention dimensions from 16 to 128, and FP32 and BF16. [Model profiling](src/lmstack/systems/monitoring.py) · [Benchmarks](benchmarks/systems)

[FlashAttention commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/1581295842c36a4c4c1f07150fe02e41d054d2a9) · [Bucketed DDP commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/98744225a5bc76e372f94a64b60fa4c3eff75933) · [Optimizer sharding commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/857e7020c1b92cd251c2c9665db76a7789d66257)

## Reasoning post-training

I built a reinforcement-learning loop for **Qwen2.5-Math-1.5B** on the **MATH** dataset. The training policy runs on one GPU and a BF16 vLLM instance generates rollouts on the other. The loop loads the current policy weights into vLLM before collecting each rollout batch. [Training loop](src/lmstack/post_training/grpo.py)

- **Rewards and advantages.** Generate multiple answers per problem, grade response format and final-answer correctness with a math grader, and compute group-relative advantages from the resulting rewards. [Reward integration](src/lmstack/post_training/grpo.py) · [Math grader](src/lmstack/post_training/grading.py)
- **Policy updates.** Compute response-token log probabilities and apply a clipped policy-ratio objective with cached old-policy log probabilities, response masking, and microbatch gradient accumulation. The runner also implements REINFORCE with and without a group baseline. [Loss implementations and loop](src/lmstack/post_training/grpo.py)
- **Experiment instrumentation.** Track mean reward, token entropy, loss, gradient norm, and clipping fraction. Held-out evaluation generates answers with vLLM and computes token entropy in minibatches. I configured a learning-rate sweep over `1e-6`, `1e-5`, and `1e-4` with a rollout batch of 256 completions and eight completions per problem. [Evaluation](src/lmstack/post_training/sft.py) · [Learning-rate experiment](https://github.com/FishDontLand/LanguageModelFromScratch/blob/ae031d63b716e50273c3c5d11ac03aa5e3812650/scripts/experiments.py)

[Rollout infrastructure commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/bdc87569170762598171cfa23446fa49b2381b51) · [GRPO commit](https://github.com/FishDontLand/LanguageModelFromScratch/commit/dd633d8bc531c6020c59868ec38e13c9fbb9a134)

## Run locally

With Python 3.11 or 3.12, install from the repository root and run the tokenizer, model, attention, and post-training loss tests.

```bash
python -m pip install -e ".[test]"
python -m pytest tests/lm tests/systems/test_attention.py tests/post_training
```

The CUDA attention tests skip without a compatible GPU. The distributed tests in `tests/systems/` use local TCP sockets for process communication. To inspect the pretraining interface, run `python -m lmstack.lm.training --help`. It takes tokenized train and validation arrays in NumPy `.npy` format.

To run clipped GRPO, use two compatible NVIDIA GPUs on Linux x86-64 and prepare the MATH train and test splits.

```bash
python -m pip install -e ".[post-training]"
python -m lmstack.post_training.prepare --data-dir data
python -c "from lmstack.post_training.grpo import run_rl; run_rl(output_dir='outputs/grpo', loss_type='grpo_clip')"
```

The runner defaults to Qwen2.5-Math-1.5B and accepts a local model checkpoint through `model_id`.

## Acknowledgments

The project follows Stanford’s Language Modeling from Scratch course. The repository retains the course starter history and upstream attribution for the math grader. [Stanford CS336](https://stanford-cs336.github.io/)
