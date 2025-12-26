import os

import numpy as np
import torch
import argparse
import wandb

from tqdm import tqdm

from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import data_loading, cross_entropy, save_checkpoint


def train_model(train_data_path: str, valid_data_path: str, batch_size: int, device: str,
                checkpoint_path: str|os.PathLike, log_freq: int,
                num_epochs: int, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int,
                context_length: int, theta: float,
                lr: float, betas: tuple, weight_decay: float, run: wandb.Run
):
    train_data = np.load(train_data_path, mmap_mode='r')
    tqdm.write("training data loaded")
    valid_data = np.load(valid_data_path, mmap_mode='r')
    tqdm.write("validation data loaded")
    lm = TransformerLM(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, theta,
                       dtype=torch.float, device=torch.device(device))
    optimizer = AdamW(lm.parameters(), lr, betas, weight_decay)
    tqdm.write("model and optimizer initialized")
    for epoch in tqdm(range(num_epochs), desc="training epoch", position=0):
        batch_inputs, batch_targets = data_loading(train_data, batch_size, context_length, device)
        optimizer.zero_grad()
        model_outputs = lm(batch_inputs)
        loss = cross_entropy(model_outputs, batch_targets)
        loss.backward()
        optimizer.step()
        save_checkpoint(lm, optimizer, epoch, checkpoint_path)
        loss_info = {"train_loss": loss.detach().cpu().item()}
        run.log(loss_info)
        if epoch % log_freq ==  log_freq - 1:
            with torch.no_grad():
                avg_valid_loss = 0
                num_valid_batches = 0
                batch_idx = 0
                num_batches = len(valid_data) // context_length
                num_iters = num_batches // batch_size
                if num_iters * batch_size < num_batches:
                    num_iters += 1
                for i in range(num_iters):
                    batch_start_idx = i * batch_size
                    batch_end_idx = min((i + 1) * batch_size, num_batches)
                    current_batch_size = batch_end_idx - batch_idx
                    input_locs = np.arange(batch_start_idx * context_length, batch_end_idx * context_length, context_length).reshape(-1, 1) + np.arange(context_length)
                    target_locs = input_locs + 1
                    valid_inputs = torch.tensor(valid_data[input_locs], device=device, dtype=torch.long)
                    valid_targets = torch.tensor(valid_data[target_locs], device=device, dtype=torch.long)
                    valid_outputs = lm(valid_inputs)
                    valid_loss = cross_entropy(valid_outputs, valid_targets)
                    ratio = current_batch_size / batch_size
                    avg_valid_loss = avg_valid_loss * num_valid_batches / (num_valid_batches + ratio) + valid_loss * ratio / (num_valid_batches + ratio)
                    num_valid_batches += ratio
                run.log({"valid_loss": avg_valid_loss.detach().cpu().item()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path to training data", type=str)
    parser.add_argument("--valid_path", help="path to validation data", type=str)
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--device", help="device to run language model", type=str)
    parser.add_argument("--checkpoint_path", help="path to save checkpoints", type=str)
    parser.add_argument("--log_freq", help="frequency of logging", type=int)
    parser.add_argument("--num_epochs", help="number of training epochs", type=int)
    parser.add_argument("--vocab_size", help="vocabulary size", type=int)
    parser.add_argument("--d_model", help="dimension of embedding", type=int)
    parser.add_argument("--d_ff", help="dimension of SwiGlu", type=int)
    parser.add_argument("--num_layers", help="number of transformer blocks", type=int)
    parser.add_argument("--num_heads", help="number of attention heads", type=int)
    parser.add_argument("--context_length", help="length of sentence", type=int)
    parser.add_argument("--theta", help="theta or RoPE", type=float)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--beta1", help="beta1 of AdamW", type=float)
    parser.add_argument("--beta2", help="beta2 of AdamW2", type=float)
    parser.add_argument("--weight_decay", help="weight decay of AdamW", type=float)

    args = parser.parse_args()

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="dyu68-georgia-institute-of-technology",
        # Set the wandb project where this run will be logged.
        project="TrainLM",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.lr,
            "architecture": "TransformerLM",
            "dataset": args.train_path,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "context_length": args.context_length,
            "theta": args.theta,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2
        },
    )


    train_model(args.train_path, args.valid_path, args.batch_size, args.device,
                args.checkpoint_path, args.log_freq,
                args.num_epochs, args.vocab_size, args.d_model, args.num_layers, args.num_heads, args.d_ff,
                args.context_length, args.theta, args.lr, (args.beta1, args.beta2), args.weight_decay, run)

    run.finish()

