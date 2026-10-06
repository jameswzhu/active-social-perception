"""Loading and windowing of ETH/UCY-style trajectory files (obsmat.txt).

File format (whitespace separated, one row per (frame, pedestrian) observation):
    frame_id  ped_id  pos_x  pos_z  pos_y  v_x  v_z  v_y
`pos_z`/`v_z` are always 0 for these datasets (2D world-plane coordinates).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class TrajectorySequence:
    """A window of `seq_len` consecutive frames containing the same set of pedestrians."""

    positions: np.ndarray  # (seq_len, num_peds, 2) world-frame x, y in meters
    ped_ids: List[int]
    source: str


def find_obsmat_files(paths: Sequence[str]) -> List[str]:
    """Expand a list of files/directories into a list of obsmat.txt file paths."""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "obsmat.txt"), recursive=True)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    if not files:
        raise FileNotFoundError(f"No obsmat.txt files found under: {paths}")
    return files


def load_obsmat(path: str) -> dict:
    """Parse an obsmat.txt file into {frame_id: {ped_id: (x, y)}}."""
    data = np.loadtxt(path)
    frame_to_peds: dict = {}
    for row in data:
        frame_id = int(round(row[0]))
        ped_id = int(round(row[1]))
        x, y = float(row[2]), float(row[4])
        frame_to_peds.setdefault(frame_id, {})[ped_id] = (x, y)
    return frame_to_peds


def build_sequences(
    frame_to_peds: dict,
    seq_len: int,
    skip: int = 1,
    min_peds: int = 1,
    source: str = "",
) -> List[TrajectorySequence]:
    """Slide a window of `seq_len` consecutive (present) frames and keep pedestrians
    that appear in every frame of the window. Windows spanning a gap in the
    frame sampling (e.g. missing annotated frames) are dropped, since they would
    mix timesteps of different durations."""
    frames_sorted = sorted(frame_to_peds.keys())
    frame_step = int(np.median(np.diff(frames_sorted))) if len(frames_sorted) > 1 else 1
    sequences: List[TrajectorySequence] = []
    for start in range(0, len(frames_sorted) - seq_len + 1, skip):
        window = frames_sorted[start : start + seq_len]
        if window[-1] - window[0] != frame_step * (seq_len - 1):
            continue
        common = set(frame_to_peds[window[0]].keys())
        for f in window[1:]:
            common &= set(frame_to_peds[f].keys())
            if not common:
                break
        if len(common) < min_peds:
            continue
        ped_ids = sorted(common)
        positions = np.array(
            [[frame_to_peds[f][pid] for pid in ped_ids] for f in window],
            dtype=np.float32,
        )
        sequences.append(TrajectorySequence(positions=positions, ped_ids=ped_ids, source=source))
    return sequences


def load_sequences(
    paths: Sequence[str],
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
    min_peds: int = 1,
) -> List[TrajectorySequence]:
    """Load one or more obsmat.txt files (or directories containing them) and build
    fixed-length (obs_len + pred_len) trajectory windows."""
    seq_len = obs_len + pred_len
    sequences: List[TrajectorySequence] = []
    for path in find_obsmat_files(paths):
        frame_to_peds = load_obsmat(path)
        sequences.extend(
            build_sequences(frame_to_peds, seq_len=seq_len, skip=skip, min_peds=min_peds, source=path)
        )
    return sequences
