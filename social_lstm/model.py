"""Social LSTM model (Alahi et al., CVPR 2016): "Social LSTM: Human Trajectory
Prediction in Crowded Spaces".

Each pedestrian is tracked by an LSTM cell with weights shared across all
pedestrians. At every timestep, a pedestrian's neighborhood is summarized by
"social pooling": the hidden states of nearby pedestrians are pooled into a
coarse spatial grid centered on the pedestrian, giving the model an implicit
notion of who is nearby and in which direction, without hand-crafted
interaction features.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def social_pooling(positions: torch.Tensor, hidden: torch.Tensor, grid_size: int, neighborhood_size: float) -> torch.Tensor:
    """Build the social hidden-state tensor for every pedestrian.

    Args:
        positions: (N, 2) current world positions.
        hidden: (N, rnn_size) hidden states from the previous timestep.
        grid_size: number of cells per side of the pooling grid (N_o in the paper).
        neighborhood_size: physical width/height (meters) of the pooling window
            centered on each pedestrian.

    Returns:
        (N, grid_size * grid_size * rnn_size) pooled tensor, flattened per pedestrian.
    """
    n, rnn_size = hidden.shape
    device = hidden.device
    num_cells = grid_size * grid_size
    if n <= 1:
        return torch.zeros(n, num_cells * rnn_size, device=device, dtype=hidden.dtype)

    # rel[i, j] = position of j relative to i
    rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)
    half = neighborhood_size / 2.0
    cell_size = neighborhood_size / grid_size

    within = (rel[..., 0].abs() < half) & (rel[..., 1].abs() < half)
    eye = torch.eye(n, dtype=torch.bool, device=device)
    within = within & ~eye

    grid_x = torch.clamp(((rel[..., 0] + half) / cell_size).floor().long(), 0, grid_size - 1)
    grid_y = torch.clamp(((rel[..., 1] + half) / cell_size).floor().long(), 0, grid_size - 1)
    cell_idx = grid_x * grid_size + grid_y  # (N, N)

    i_idx, j_idx = within.nonzero(as_tuple=True)
    pooled_flat = torch.zeros(n * num_cells, rnn_size, device=device, dtype=hidden.dtype)
    if i_idx.numel() > 0:
        dest = i_idx * num_cells + cell_idx[i_idx, j_idx]
        src = hidden[j_idx]
        pooled_flat.index_add_(0, dest, src)
    return pooled_flat.view(n, num_cells * rnn_size)


class SocialLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        rnn_size: int = 128,
        grid_size: int = 8,
        neighborhood_size: float = 2.0,
    ):
        super().__init__()
        self.rnn_size = rnn_size
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size

        self.input_embedding = nn.Linear(2, embedding_dim)
        self.social_embedding = nn.Linear(grid_size * grid_size * rnn_size, embedding_dim)
        self.cell = nn.LSTMCell(2 * embedding_dim, rnn_size)
        self.output_layer = nn.Linear(rnn_size, 5)

    def init_hidden(self, num_peds: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        h = torch.zeros(num_peds, self.rnn_size, device=device)
        c = torch.zeros(num_peds, self.rnn_size, device=device)
        return h, c

    def step(
        self,
        positions_t: torch.Tensor,
        prev_positions: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        """Advance one timestep for all pedestrians present in the scene.

        The LSTM input is the displacement since the previous timestep and the
        predicted mean is an offset from the current position, so the model
        never has to regress absolute scene coordinates.

        Args:
            positions_t: (N, 2) positions at time t (used to build the social tensor).
            prev_positions: (N, 2) positions at time t-1 (equal to positions_t at t=0).
            h, c: (N, rnn_size) previous hidden/cell states.

        Returns:
            gaussian params (mux, muy, sx, sy, corr), each (N, 1); new (h, c).
        """
        pooled = social_pooling(positions_t, h, self.grid_size, self.neighborhood_size)
        e = F.relu(self.input_embedding(positions_t - prev_positions))
        a = F.relu(self.social_embedding(pooled))
        x = torch.cat([e, a], dim=-1)
        h, c = self.cell(x, (h, c))
        out = self.output_layer(h)
        dx, dy, logsx, logsy, corr_raw = out.split(1, dim=-1)
        mux = positions_t[:, 0:1] + dx
        muy = positions_t[:, 1:2] + dy
        sx = torch.exp(torch.clamp(logsx, -10, 10))
        sy = torch.exp(torch.clamp(logsy, -10, 10))
        corr = torch.tanh(corr_raw) * 0.999
        return (mux, muy, sx, sy, corr), h, c


def gaussian_nll(mux, muy, sx, sy, corr, x, y, eps: float = 1e-6) -> torch.Tensor:
    """Negative log-likelihood of a bivariate Gaussian, per pedestrian (N,)."""
    normx = x - mux.squeeze(-1)
    normy = y - muy.squeeze(-1)
    sx = sx.squeeze(-1)
    sy = sy.squeeze(-1)
    corr = corr.squeeze(-1)

    one_minus_rho2 = torch.clamp(1 - corr ** 2, min=eps)
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * corr * normx * normy / (sx * sy)
    log_denom = torch.log(2 * math.pi * sx * sy * torch.sqrt(one_minus_rho2) + eps)
    return z / (2 * one_minus_rho2) + log_denom


def squared_mahalanobis(mux, muy, sx, sy, corr, x, y, eps: float = 1e-6) -> torch.Tensor:
    """Squared Mahalanobis distance d_MD^2 = (X-mu)^T Sigma^-1 (X-mu), per pedestrian (N,).

    Under an ideal (correctly calibrated) bivariate Gaussian this quantity is
    distributed as chi-squared with 2 degrees of freedom.
    """
    normx = x - mux.squeeze(-1)
    normy = y - muy.squeeze(-1)
    sx = sx.squeeze(-1)
    sy = sy.squeeze(-1)
    corr = corr.squeeze(-1)

    one_minus_rho2 = torch.clamp(1 - corr ** 2, min=eps)
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * corr * normx * normy / (sx * sy)
    return z / one_minus_rho2
