"""Train a Social LSTM trajectory predictor on the UCY Zara dataset and
visualize its predictions on a small held-out sample.

Example:
    python main.py --epochs 50 --loss calibrated --beta 1.0 --num-vis 6

The model observes a variable number of steps (--min-obs-len..--max-obs-len,
2..12 by default) and always predicts --pred-len steps. Windows hold
max_obs_len + pred_len frames; a sequence with k observed steps uses the last
k + pred_len frames, so the prediction targets are the same for every k.

By default this trains on both Zara scenes (data/ucy/zara01, data/ucy/zara02),
splitting sequences into train/validation/a small visualization sample. After
training, it saves a checkpoint and PNG plots (observed / ground-truth /
predicted trajectories) for a few sample sequences.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from social_lstm.dataset import load_sequences
from social_lstm.engine import (
    batch_calibrated_loss,
    delta_esv,
    displacement_errors,
    predict_sequence,
    predict_sequence_with_gaussian,
    sequence_nll,
)
from social_lstm.model import SocialLSTM, squared_mahalanobis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data",
        nargs="+",
        default=["data/ucy/zara01", "data/ucy/zara02"],
        help="obsmat.txt files or directories to search (default: full Zara dataset)",
    )
    p.add_argument("--min-obs-len", type=int, default=2, help="fewest observed steps sampled during training")
    p.add_argument("--max-obs-len", type=int, default=12, help="most observed steps sampled during training")
    p.add_argument(
        "--eval-obs-lens",
        type=int,
        nargs="+",
        default=[2, 4, 8, 12],
        help="observation lengths to evaluate; the best checkpoint minimizes their mean val ADE",
    )
    p.add_argument("--vis-obs-len", type=int, default=8, help="observation length used for the plots")
    p.add_argument("--pred-len", type=int, default=12)
    p.add_argument("--skip", type=int, default=4, help="stride (in frames) between sampled windows")
    p.add_argument("--min-peds", type=int, default=1, help="min pedestrians required per window")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.05, help="held-out fraction used only for visualization")
    p.add_argument("--num-vis", type=int, default=6, help="number of test sequences to plot")
    p.add_argument(
        "--uncertainty-sigma",
        type=int,
        nargs="+",
        default=[1, 2],
        help="k-sigma confidence ellipses to draw around each predicted position",
    )
    p.add_argument("--uncertainty-alpha", type=float, default=0.15, help="fill transparency of uncertainty ellipses")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8, help="sequences per gradient step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--loss", choices=["nll", "calibrated"], default="calibrated")
    p.add_argument("--beta", type=float, default=1.0, help="weight of the calibration CDF term (calibrated loss only)")
    p.add_argument("--kde-bandwidth", type=float, default=0.05)
    p.add_argument("--kde-bin-step", type=float, default=0.01)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--rnn-size", type=int, default=128)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--neighborhood-size", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-every", type=int, default=1, help="run validation every N epochs")
    p.add_argument("--save", default="checkpoints/social_lstm_zara.pt")
    p.add_argument("--plot-dir", default="plots/zara")
    return p.parse_args()


def split_sequences(sequences, val_frac, test_frac):
    n_test = max(1, int(len(sequences) * test_frac))
    n_val = max(1, int(len(sequences) * val_frac))
    test_seqs = sequences[:n_test]
    val_seqs = sequences[n_test : n_test + n_val]
    train_seqs = sequences[n_test + n_val :]
    return train_seqs, val_seqs, test_seqs


def crop(positions, obs_len, max_obs_len):
    """Keep the last obs_len + pred_len frames of a max_obs_len + pred_len window."""
    return positions[max_obs_len - obs_len :]


def train(model, optimizer, train_seqs, val_seqs, args, device):
    """Train for `args.epochs` epochs, then restore the weights with the lowest
    validation ADE (validation ADE fluctuates noticeably between epochs)."""
    best_ade, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_seqs)
        epoch_loss, n_batches = 0.0, 0
        start = time.time()
        for batch_start in range(0, len(train_seqs), args.batch_size):
            batch = train_seqs[batch_start : batch_start + args.batch_size]
            obs_lens = [random.randint(args.min_obs_len, args.max_obs_len) for _ in batch]
            batch_positions = [
                crop(torch.from_numpy(seq.positions).to(device), k, args.max_obs_len)
                for seq, k in zip(batch, obs_lens)
            ]
            optimizer.zero_grad()
            if args.loss == "nll":
                loss = torch.stack([sequence_nll(model, p) for p in batch_positions]).mean()
            else:
                loss = batch_calibrated_loss(
                    model,
                    batch_positions,
                    obs_len=obs_lens,
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
            results = evaluate_obs_lens(model, val_seqs, args, device)
            ade = sum(r[0] for r in results.values()) / len(results)
            fde = sum(r[1] for r in results.values()) / len(results)
            msg += f"  val_ADE={ade:.3f}  val_FDE={fde:.3f}  (ADE by obs_len: "
            msg += " ".join(f"{k}:{a:.3f}" for k, (a, _) in results.items()) + ")"
            if ade < best_ade:
                best_ade, best_epoch = ade, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best checkpoint from epoch {best_epoch} (val_ADE={best_ade:.3f})")


@torch.no_grad()
def evaluate_obs_lens(model, sequences, args, device):
    """{obs_len: (ADE, FDE)} for every length in args.eval_obs_lens."""
    return {
        k: evaluate(model, sequences, k, args.max_obs_len, args.pred_len, device)
        for k in args.eval_obs_lens
    }


@torch.no_grad()
def evaluate(model, sequences, obs_len, max_obs_len, pred_len, device):
    model.eval()
    ades, fdes = [], []
    for seq in sequences:
        positions = crop(torch.from_numpy(seq.positions).to(device), obs_len, max_obs_len)
        pred = predict_sequence(model, positions, obs_len, pred_len)
        target = positions[obs_len:]
        ade, fde = displacement_errors(pred, target)
        ades.append(ade)
        fdes.append(fde)
    return sum(ades) / len(ades), sum(fdes) / len(fdes)


@torch.no_grad()
def visualize(model, test_seqs, args, device):
    os.makedirs(args.plot_dir, exist_ok=True)
    model.eval()

    maha_all = []
    for i, seq in enumerate(test_seqs[: args.num_vis]):
        obs_len = args.vis_obs_len
        positions = crop(torch.from_numpy(seq.positions).to(device), obs_len, args.max_obs_len)
        obs = positions[:obs_len].cpu().numpy()
        gt_future = positions[obs_len:].cpu().numpy()
        pred_future, sigma_x, sigma_y, corr = predict_sequence_with_gaussian(
            model, positions, obs_len, args.pred_len
        )
        target = positions[obs_len:]
        maha = squared_mahalanobis(
            pred_future[..., 0:1], pred_future[..., 1:2], sigma_x.unsqueeze(-1), sigma_y.unsqueeze(-1),
            corr.unsqueeze(-1), target[..., 0], target[..., 1],
        )
        maha_all.append(maha)
        pred_future_np = pred_future.cpu().numpy()
        sigma_x_np, sigma_y_np, corr_np = sigma_x.cpu().numpy(), sigma_y.cpu().numpy(), corr.cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        num_peds = obs.shape[1]
        colors = plt.cm.tab10(torch.linspace(0, 1, max(num_peds, 2)).numpy())
        for p in range(num_peds):
            color = colors[p % len(colors)]
            for t in range(args.pred_len):
                for k in args.uncertainty_sigma:
                    ax.add_patch(
                        covariance_ellipse(
                            pred_future_np[t, p, 0],
                            pred_future_np[t, p, 1],
                            sigma_x_np[t, p],
                            sigma_y_np[t, p],
                            corr_np[t, p],
                            k=k,
                            color=color,
                            alpha=args.uncertainty_alpha,
                        )
                    )
            ax.plot(obs[:, p, 0], obs[:, p, 1], "-o", color=color, markersize=3, label="observed" if p == 0 else None)
            ax.plot(
                [obs[-1, p, 0], *gt_future[:, p, 0]],
                [obs[-1, p, 1], *gt_future[:, p, 1]],
                "--o",
                color=color,
                markersize=3,
                label="ground truth" if p == 0 else None,
            )
            ax.plot(
                [obs[-1, p, 0], *pred_future_np[:, p, 0]],
                [obs[-1, p, 1], *pred_future_np[:, p, 1]],
                ":x",
                color=color,
                markersize=4,
                label="predicted" if p == 0 else None,
            )
        sigma_label = "/".join(f"{k}σ" for k in args.uncertainty_sigma)
        ax.plot([], [], color="gray", alpha=0.4, linewidth=8, label=f"{sigma_label} uncertainty")
        ax.set_title(f"Zara test sample {i}  ({num_peds} pedestrians, {obs_len} observed steps)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend()
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        out_path = os.path.join(args.plot_dir, f"sample_{i}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    if maha_all:
        esv = delta_esv(torch.cat([m.reshape(-1) for m in maha_all]))
        print(
            f"test sample calibration: ESV1={esv['esv1']:+.3f}  ESV2={esv['esv2']:+.3f}  "
            f"ESV3={esv['esv3']:+.3f}  |ESV|_mean={esv['mean_abs_esv']:.3f}"
        )


def covariance_ellipse(cx, cy, sx, sy, corr, k, **kwargs):
    """A matplotlib Ellipse patch enclosing the k-sigma confidence region of a
    bivariate Gaussian with the given marginal std-devs and correlation."""
    cov = np.array([[sx ** 2, corr * sx * sy], [corr * sx * sy, sy ** 2]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * k * np.sqrt(np.clip(eigvals, 0, None))
    return Ellipse((cx, cy), width=width, height=height, angle=angle, **kwargs)


def save_checkpoint(model, args):
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
            "obs_len": args.max_obs_len,
            "min_obs_len": args.min_obs_len,
            "max_obs_len": args.max_obs_len,
            "pred_len": args.pred_len,
            "loss": args.loss,
        },
        args.save,
    )
    print(f"Saved checkpoint to {args.save}")


def main():
    args = parse_args()
    if not 2 <= args.min_obs_len <= args.max_obs_len:
        raise ValueError("need 2 <= --min-obs-len <= --max-obs-len (the model needs a velocity to start from)")
    if any(not 2 <= k <= args.max_obs_len for k in args.eval_obs_lens + [args.vis_obs_len]):
        raise ValueError("--eval-obs-lens and --vis-obs-len must lie in [2, --max-obs-len]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sequences = load_sequences(
        args.data, obs_len=args.max_obs_len, pred_len=args.pred_len, skip=args.skip, min_peds=args.min_peds
    )
    if not sequences:
        raise RuntimeError("No trajectory windows were built from the given data (try a smaller --skip).")
    random.shuffle(sequences)
    train_seqs, val_seqs, test_seqs = split_sequences(sequences, args.val_frac, args.test_frac)
    print(
        f"Loaded {len(sequences)} sequences "
        f"({len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test)"
    )

    device = torch.device(args.device)
    model = SocialLSTM(
        embedding_dim=args.embedding_dim,
        rnn_size=args.rnn_size,
        grid_size=args.grid_size,
        neighborhood_size=args.neighborhood_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, train_seqs, val_seqs, args, device)
    save_checkpoint(model, args)

    if test_seqs:
        for k, (test_ade, test_fde) in evaluate_obs_lens(model, test_seqs, args, device).items():
            print(f"obs_len={k:2d}  test_ADE={test_ade:.3f}  test_FDE={test_fde:.3f}")
        visualize(model, test_seqs, args, device)


if __name__ == "__main__":
    main()
