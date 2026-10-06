"""Shared routines for running the Social LSTM over a full trajectory sequence:
teacher-forced loss computation (training) and autoregressive rollout (inference).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch

from .calibration_loss import calibrated_loss
from .model import SocialLSTM, gaussian_nll, squared_mahalanobis


def sequence_nll(model: SocialLSTM, positions: torch.Tensor) -> torch.Tensor:
    """Teacher-forced negative log-likelihood loss for one sequence.

    Args:
        positions: (seq_len, num_peds, 2) ground-truth positions for every
            pedestrian present throughout the whole window.

    Returns:
        scalar mean NLL over all predicted timesteps and pedestrians.
    """
    seq_len, num_peds, _ = positions.shape
    device = positions.device
    h, c = model.init_hidden(num_peds, device=device)

    losses = []
    for t in range(seq_len - 1):
        (mux, muy, sx, sy, corr), h, c = model.step(positions[t], positions[max(t - 1, 0)], h, c)
        target = positions[t + 1]
        nll = gaussian_nll(mux, muy, sx, sy, corr, target[:, 0], target[:, 1])
        losses.append(nll)
    return torch.stack(losses).mean()


def _rollout_prediction_horizon(model: SocialLSTM, positions: torch.Tensor, obs_len: int):
    """Observe the first `obs_len` ground-truth positions, then roll out the
    prediction horizon autoregressively (feeding back predicted means, as at
    inference), collecting Gaussian outputs only for t = obs_len .. seq_len-1.
    This matches Eq. 2 of the calibration paper, which sums the loss over
    T_obs+1..T_pred, and makes the predicted covariance account for error
    accumulated over the rollout rather than only one-step error.
    """
    seq_len, num_peds, _ = positions.shape
    device = positions.device
    h, c = model.init_hidden(num_peds, device=device)

    d_list, mu_list, target_list = [], [], []
    prev_pos = cur_pos = positions[0]
    for t in range(seq_len - 1):
        (mux, muy, sx, sy, corr), h, c = model.step(cur_pos, prev_pos, h, c)
        mu = torch.cat([mux, muy], dim=-1)
        if t >= obs_len - 1:
            target = positions[t + 1]
            d = squared_mahalanobis(mux, muy, sx, sy, corr, target[:, 0], target[:, 1])
            d_list.append(d)
            mu_list.append(mu)
            target_list.append(target)
            prev_pos, cur_pos = cur_pos, mu
        else:
            prev_pos, cur_pos = cur_pos, positions[t + 1]
    return torch.cat(d_list), torch.cat(mu_list, dim=0), torch.cat(target_list, dim=0)


def batch_calibrated_loss(
    model: SocialLSTM,
    batch_positions: Sequence[torch.Tensor],
    obs_len: Union[int, Sequence[int]],
    beta: float = 1.0,
    bandwidth: float = 0.05,
    bin_step: float = 0.01,
) -> torch.Tensor:
    """Calibrated loss (Eq. 5) pooled over every sequence in the batch, so the
    KDE-based CDF matching term has enough Mahalanobis-distance samples.

    `obs_len` is either one observation length for the whole batch or one per
    sequence (each sequence then holds obs_len[i] + pred_len frames)."""
    obs_lens = [obs_len] * len(batch_positions) if isinstance(obs_len, int) else obs_len
    d_all, mu_all, target_all = [], [], []
    for positions, n_obs in zip(batch_positions, obs_lens):
        d, mu, target = _rollout_prediction_horizon(model, positions, n_obs)
        d_all.append(d)
        mu_all.append(mu)
        target_all.append(target)
    d_all = torch.cat(d_all)
    mu_all = torch.cat(mu_all, dim=0)
    target_all = torch.cat(target_all, dim=0)
    return calibrated_loss(d_all, mu_all, target_all, beta=beta, bandwidth=bandwidth, bin_step=bin_step)


@torch.no_grad()
def predict_sequence(
    model: SocialLSTM,
    positions: torch.Tensor,
    obs_len: int,
    pred_len: int,
    sample: bool = False,
) -> torch.Tensor:
    """Observe the first `obs_len` ground-truth positions, then roll out
    `pred_len` future positions autoregressively.

    Returns:
        (pred_len, num_peds, 2) predicted positions.
    """
    device = positions.device
    num_peds = positions.shape[1]
    h, c = model.init_hidden(num_peds, device=device)

    prev_pos = cur_pos = positions[0]
    for t in range(obs_len - 1):
        (mux, muy, sx, sy, corr), h, c = model.step(positions[t], positions[max(t - 1, 0)], h, c)
        prev_pos, cur_pos = positions[t], positions[t + 1]

    predictions: List[torch.Tensor] = []
    for _ in range(pred_len):
        (mux, muy, sx, sy, corr), h, c = model.step(cur_pos, prev_pos, h, c)
        prev_pos = cur_pos
        if sample:
            cur_pos = _sample_gaussian(mux, muy, sx, sy, corr)
        else:
            cur_pos = torch.cat([mux, muy], dim=-1)
        predictions.append(cur_pos)
    return torch.stack(predictions, dim=0)


@torch.no_grad()
def predict_sequence_with_uncertainty(
    model: SocialLSTM,
    positions: torch.Tensor,
    obs_len: int,
    pred_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like `predict_sequence`, but also returns the squared Mahalanobis
    distance of the ground truth under each predicted Gaussian, for calibration
    evaluation (Delta-ESV metrics).

    Returns:
        pred: (pred_len, num_peds, 2) predicted means.
        maha: (pred_len, num_peds) squared Mahalanobis distance to ground truth.
    """
    device = positions.device
    num_peds = positions.shape[1]
    h, c = model.init_hidden(num_peds, device=device)

    prev_pos = cur_pos = positions[0]
    for t in range(obs_len - 1):
        _, h, c = model.step(positions[t], positions[max(t - 1, 0)], h, c)
        prev_pos, cur_pos = positions[t], positions[t + 1]

    preds, mahas = [], []
    for t in range(pred_len):
        (mux, muy, sx, sy, corr), h, c = model.step(cur_pos, prev_pos, h, c)
        target = positions[obs_len + t]
        d = squared_mahalanobis(mux, muy, sx, sy, corr, target[:, 0], target[:, 1])
        prev_pos, cur_pos = cur_pos, torch.cat([mux, muy], dim=-1)
        preds.append(cur_pos)
        mahas.append(d)
    return torch.stack(preds, dim=0), torch.stack(mahas, dim=0)


@torch.no_grad()
def predict_sequence_with_gaussian(
    model: SocialLSTM,
    positions: torch.Tensor,
    obs_len: int,
    pred_len: int,
):
    """Like `predict_sequence`, but also returns the full predicted Gaussian
    covariance at each rollout step, for plotting uncertainty regions.

    Returns:
        pred: (pred_len, num_peds, 2) predicted means.
        sigma_x, sigma_y, corr: (pred_len, num_peds) predicted covariance parameters.
    """
    device = positions.device
    num_peds = positions.shape[1]
    h, c = model.init_hidden(num_peds, device=device)

    prev_pos = cur_pos = positions[0]
    for t in range(obs_len - 1):
        _, h, c = model.step(positions[t], positions[max(t - 1, 0)], h, c)
        prev_pos, cur_pos = positions[t], positions[t + 1]

    preds, sxs, sys_, corrs = [], [], [], []
    for _ in range(pred_len):
        (mux, muy, sx, sy, corr), h, c = model.step(cur_pos, prev_pos, h, c)
        prev_pos, cur_pos = cur_pos, torch.cat([mux, muy], dim=-1)
        preds.append(cur_pos)
        sxs.append(sx.squeeze(-1))
        sys_.append(sy.squeeze(-1))
        corrs.append(corr.squeeze(-1))
    return (
        torch.stack(preds, dim=0),
        torch.stack(sxs, dim=0),
        torch.stack(sys_, dim=0),
        torch.stack(corrs, dim=0),
    )


def delta_esv(maha: torch.Tensor, sigma_levels=(1, 2, 3)) -> dict:
    """Delta Empirical Sigma Value: empirical proportion of ground truth points
    falling within the predicted k-sigma confidence region, minus the
    proportion expected under an ideal bivariate Gaussian, 1 - exp(-k^2/2).

    Negative values indicate over-confident predictions (region too small);
    positive values indicate under-confident predictions (region too large).
    """
    d = maha.reshape(-1)
    result = {}
    for k in sigma_levels:
        ideal = 1.0 - math.exp(-0.5 * k ** 2)
        empirical = (d <= k ** 2).float().mean().item()
        result[f"esv{k}"] = empirical - ideal
    result["mean_abs_esv"] = sum(abs(v) for v in result.values()) / len(result)
    return result


def _sample_gaussian(mux, muy, sx, sy, corr) -> torch.Tensor:
    mean = torch.cat([mux, muy], dim=-1)
    sx, sy, corr = sx.squeeze(-1), sy.squeeze(-1), corr.squeeze(-1)
    cov_xy = corr * sx * sy
    cov = torch.stack(
        [torch.stack([sx ** 2, cov_xy], dim=-1), torch.stack([cov_xy, sy ** 2], dim=-1)], dim=-2
    )
    dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
    return dist.sample()


def displacement_errors(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """ADE (mean over all timesteps) and FDE (final timestep only), averaged over pedestrians."""
    dist = torch.norm(pred - target, dim=-1)  # (pred_len, num_peds)
    ade = dist.mean().item()
    fde = dist[-1].mean().item()
    return ade, fde
