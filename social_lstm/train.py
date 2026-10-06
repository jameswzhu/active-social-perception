"""Train a Social LSTM trajectory predictor on ETH/UCY-style data.

Supports two loss functions:
  --loss nll         Standard negative log-likelihood (Social LSTM, Alahi et al. 2016).
  --loss calibrated  Calibration loss from "Rethinking Gaussian Trajectory
                      Predictors: Calibrated Uncertainty for Safe Planning"
                      (Cheraghi Pouria et al., arXiv:2603.10407), which matches
                      the empirical distribution of squared Mahalanobis
                      distances to a chi^2_2 distribution via KDE, plus a mean
                      position error term.

Example:
    python -m social_lstm.train --data data/ucy/zara01 data/ucy/zara02 \\
        --loss calibrated --beta 1.0 --epochs 20 --batch-size 8 \\
        --save checkpoints/social_lstm_calibrated.pt
"""
from __future__ import annotations

import argparse
import random
import time

import torch

from .dataset import load_sequences
from .engine import sequence_nll, batch_calibrated_loss, predict_sequence, displacement_errors
from .model import SocialLSTM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="obsmat.txt files or directories to search")
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--skip", type=int, default=4, help="stride (in frames) between sampled windows")
    p.add_argument("--min-peds", type=int, default=1, help="min pedestrians required per window")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="sequences per gradient step")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--loss", choices=["nll", "calibrated"], default="nll")
    p.add_argument("--beta", type=float, default=1.0, help="weight of the calibration CDF term (calibrated loss only)")
    p.add_argument("--kde-bandwidth", type=float, default=0.05)
    p.add_argument("--kde-bin-step", type=float, default=0.01)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--rnn-size", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--neighborhood-size", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", default="checkpoints/social_lstm.pt")
    p.add_argument("--eval-every", type=int, default=1, help="run validation every N epochs")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sequences = load_sequences(
        args.data, obs_len=args.obs_len, pred_len=args.pred_len, skip=args.skip, min_peds=args.min_peds
    )
    if not sequences:
        raise RuntimeError("No trajectory windows were built from the given data (try a smaller --skip).")
    random.shuffle(sequences)
    n_val = max(1, int(len(sequences) * args.val_frac))
    val_seqs, train_seqs = sequences[:n_val], sequences[n_val:]
    print(f"Loaded {len(sequences)} sequences ({len(train_seqs)} train / {len(val_seqs)} val)")

    device = torch.device(args.device)
    model = SocialLSTM(
        embedding_dim=args.embedding_dim,
        rnn_size=args.rnn_size,
        grid_size=args.grid_size,
        neighborhood_size=args.neighborhood_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_seqs)
        epoch_loss, n_batches = 0.0, 0
        start = time.time()
        for batch_start in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[batch_start : batch_start + args.batch_size]
            batch_positions = [torch.from_numpy(seq.positions).to(device) for seq in batch]
            optimizer.zero_grad()
            if args.loss == "nll":
                loss = torch.stack([sequence_nll(model, p) for p in batch_positions]).mean()
            else:
                loss = batch_calibrated_loss(
                    model,
                    batch_positions,
                    obs_len=args.obs_len,
                    beta=args.beta,
                    bandwidth=args.kde_bandwidth,
                    bin_step=args.kde_bin_step,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        msg = f"epoch {epoch:3d}/{args.epochs}  train_{args.loss}={avg_loss:.4f}  ({time.time() - start:.1f}s)"

        if val_seqs and epoch % args.eval_every == 0:
            ade, fde = evaluate(model, val_seqs, args.obs_len, args.pred_len, device)
            msg += f"  val_ADE={ade:.3f}  val_FDE={fde:.3f}"
        print(msg)

    save_checkpoint(model, args)


@torch.no_grad()
def evaluate(model, sequences, obs_len, pred_len, device):
    model.eval()
    ades, fdes = [], []
    for seq in sequences:
        positions = torch.from_numpy(seq.positions).to(device)
        pred = predict_sequence(model, positions, obs_len, pred_len)
        target = positions[obs_len:]
        ade, fde = displacement_errors(pred, target)
        ades.append(ade)
        fdes.append(fde)
    return sum(ades) / len(ades), sum(fdes) / len(fdes)


def save_checkpoint(model, args):
    import os

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparams": {
                "embedding_dim": args.embedding_dim,
                "rnn_size": args.rnn_size,
                "grid_size": args.grid_size,
                "neighborhood_size": args.neighborhood_size,
            },
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "loss": args.loss,
        },
        args.save,
    )
    print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
