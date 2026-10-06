"""Evaluate a trained Social LSTM checkpoint: ADE/FDE on held-out trajectory windows.

Example:
    python -m social_lstm.evaluate --data data/ucy/zara01 --checkpoint checkpoints/social_lstm.pt
"""
from __future__ import annotations

import argparse

import torch

from .dataset import load_sequences
from .engine import predict_sequence, displacement_errors, predict_sequence_with_uncertainty, delta_esv
from .model import SocialLSTM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="obsmat.txt files or directories to search")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--obs-len",
        type=int,
        default=None,
        help="number of observed steps (default: the checkpoint's obs_len)",
    )
    p.add_argument("--skip", type=int, default=4)
    p.add_argument("--min-peds", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=1, help="stochastic rollouts per sequence (best-of-N FDE/ADE)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    obs_len = args.obs_len if args.obs_len is not None else ckpt["obs_len"]
    pred_len = ckpt["pred_len"]

    model = SocialLSTM(**ckpt["hyperparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sequences = load_sequences(
        args.data, obs_len=obs_len, pred_len=pred_len, skip=args.skip, min_peds=args.min_peds
    )
    if not sequences:
        raise RuntimeError("No trajectory windows were built from the given data.")

    ades, fdes = [], []
    maha_all = []
    with torch.no_grad():
        for seq in sequences:
            positions = torch.from_numpy(seq.positions).to(device)
            target = positions[obs_len:]
            best_ade, best_fde = None, None
            for _ in range(args.num_samples):
                pred = predict_sequence(
                    model, positions, obs_len, pred_len, sample=args.num_samples > 1
                )
                ade, fde = displacement_errors(pred, target)
                if best_ade is None or ade < best_ade:
                    best_ade, best_fde = ade, fde
            ades.append(best_ade)
            fdes.append(best_fde)

            # Calibration is evaluated on the deterministic (mean) rollout, since it
            # measures the reliability of the predicted covariance itself.
            _, maha = predict_sequence_with_uncertainty(model, positions, obs_len, pred_len)
            maha_all.append(maha)

    esv = delta_esv(torch.cat([m.reshape(-1) for m in maha_all]))
    print(
        f"obs_len={obs_len}  sequences={len(sequences)}  ADE={sum(ades) / len(ades):.4f}  FDE={sum(fdes) / len(fdes):.4f}  "
        f"ESV1={esv['esv1']:+.3f}  ESV2={esv['esv2']:+.3f}  ESV3={esv['esv3']:+.3f}  "
        f"|ESV|_mean={esv['mean_abs_esv']:.3f}"
    )


if __name__ == "__main__":
    main()
