"""Calibration loss from "Rethinking Gaussian Trajectory Predictors: Calibrated
Uncertainty for Safe Planning" (Cheraghi Pouria, Golchoubian, Driggs-Campbell,
arXiv:2603.10407).

Standard NLL training of a bivariate Gaussian predictor tends to produce
over- or under-confident uncertainty estimates because the NLL loss rewards
driving the squared Mahalanobis distance towards zero, whereas an "ideal"
bivariate Gaussian predictor has squared Mahalanobis distances between its
predicted mean/covariance and the true position that follow a Chi-squared
distribution with 2 degrees of freedom (chi^2_2), not a degenerate spike at 0.

This module implements the paper's model-agnostic replacement loss:
  1. Estimate the empirical distribution of squared Mahalanobis distances via
     a differentiable Kernel Density Estimate (Eq. 3).
  2. Penalize the mean-absolute deviation between the estimated CDF and the
     theoretical chi^2_2 CDF (L_CDF, Eq. 4).
  3. Add a mean-absolute-error term on the predicted mean to keep point
     accuracy in check, since L_CDF alone only constrains the covariance (Eq. 5).

Note on aggregation: the paper indexes Mahalanobis samples by (timestep t,
pedestrian i). A single timestep in one scene often has very few pedestrians,
which is too small a sample for a reliable KDE, so here the empirical
distribution is estimated by pooling samples across every prediction timestep
and every pedestrian in a training batch. This is a practical batching choice
that preserves the loss's meaning (matching the pooled empirical distribution
of confidence levels to chi^2_2) while giving the KDE step enough samples.
"""
from __future__ import annotations

import torch


def chi2_2_cdf(x: torch.Tensor) -> torch.Tensor:
    """Closed-form CDF of a chi-squared distribution with 2 degrees of freedom."""
    return 1.0 - torch.exp(-0.5 * x)


def cdf_matching_loss(
    squared_mahalanobis: torch.Tensor,
    bandwidth: float = 0.05,
    bin_step: float = 0.01,
    max_range: float = 20.0,
) -> torch.Tensor:
    """L_CDF (Eq. 3-4): mean absolute error between the KDE-estimated CDF of
    squared Mahalanobis distances and the theoretical chi^2_2 CDF.

    The KDE is evaluated on a fixed grid of bin centers over [0, max_range)
    (chi^2_2 has ~99.995% of its mass below 20) and normalized by its sum, so
    samples beyond the grid show up as missing CDF mass instead of stretching
    the grid. A softmax normalization with a small temperature is avoided:
    with T ~ 0.005 it puts all mass on a single bin, which turns the CDF into
    a step function and makes over-confident predictions score better than
    calibrated ones.

    Args:
        squared_mahalanobis: (M,) flattened squared Mahalanobis distances
            pooled over pedestrians/timesteps.
        bandwidth: KDE kernel bandwidth `b`.
        bin_step: spacing between histogram bin centers `c_j`.
        max_range: upper end of the bin grid.
    """
    d = squared_mahalanobis.reshape(-1)
    bin_centers = torch.arange(0.0, max_range, bin_step, device=d.device, dtype=d.dtype)

    # Eq. 3: Gaussian-kernel density estimate at each bin center.
    diff = (bin_centers.unsqueeze(1) - d.unsqueeze(0)) / bandwidth  # (num_bins, M)
    density = torch.exp(-0.5 * diff ** 2).sum(dim=1)

    # Normalize over the full (unclipped) sample count: distances beyond the
    # grid leave the estimated CDF below 1, which the loss then penalizes.
    probs = density / (d.numel() * bandwidth * (2 * torch.pi) ** 0.5 / bin_step)
    cdf_hat = torch.cumsum(probs, dim=0)

    cdf_theoretical = chi2_2_cdf(bin_centers)
    return torch.mean(torch.abs(cdf_hat - cdf_theoretical))


def calibrated_loss(
    squared_mahalanobis: torch.Tensor,
    mean: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    bandwidth: float = 0.05,
    bin_step: float = 0.01,
) -> torch.Tensor:
    """Full loss of Eq. 5: L = beta * L_CDF + mean absolute position error.

    Args:
        squared_mahalanobis: (M,) pooled squared Mahalanobis distances.
        mean: (M, 2) predicted Gaussian means.
        target: (M, 2) ground-truth positions.
        beta: weight of the calibration term relative to the mean-error term.
    """
    cdf_loss = cdf_matching_loss(squared_mahalanobis, bandwidth, bin_step)
    mean_error = torch.norm(mean - target, dim=-1).mean()
    return beta * cdf_loss + mean_error
