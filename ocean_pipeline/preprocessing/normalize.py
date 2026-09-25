"""
Phase 0 — Step 5: Normalization utilities.

Method: z-score per channel (µ = 0, σ = 1) computed across the training period.
Rationale: z-score is preferred over min-max because:
  1. Ocean variables (especially temperature at depth) have heavy tails.
  2. Z-score is more robust to outliers that appear in sparse-coverage products.
  3. New data outside the training value range doesn't saturate to 0 or 1.

Stats are computed once from the training data and saved to:
    data/masks/norm_stats.npz

Format of norm_stats.npz:
    channel_names: list of channel names in order
    means: array [7]   — per-channel mean over training ocean cells
    stds:  array [7]   — per-channel std over training ocean cells
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from typing import Optional

from ocean_pipeline.config import INPUT_CHANNELS, DATA_MASKS


NORM_STATS_PATH = DATA_MASKS / "norm_stats.npz"


def load_norm_stats(path: Path = NORM_STATS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-computed normalisation stats. Returns (means, stds)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {path}. "
            "Run build_norm_stats() on training data first."
        )
    data = np.load(path)
    return data["means"], data["stds"]


def build_norm_stats(tensors: np.ndarray,
                     ocean_mask: Optional[np.ndarray] = None,
                     channel_names: list = INPUT_CHANNELS,
                     save_path: Path = NORM_STATS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute and save per-channel z-score statistics from training tensors.

    Parameters
    ----------
    tensors : np.ndarray, shape (N_days, H, W, C) — stacked training tensors
    ocean_mask : bool array [H, W] — if provided, statistics computed over ocean only
    channel_names : list of channel names (must match last dim of tensors)
    save_path : where to save the .npz

    Returns (means, stds) each shape (C,)
    """
    N, H, W, C = tensors.shape
    assert C == len(channel_names), f"Expected {len(channel_names)} channels, got {C}"

    means = np.zeros(C, dtype=np.float64)
    stds  = np.zeros(C, dtype=np.float64)

    for c in range(C):
        data = tensors[:, :, :, c]   # (N, H, W)
        if ocean_mask is not None:
            # Only ocean cells
            data = data[:, ocean_mask]   # (N, n_ocean)
        valid = data[~np.isnan(data)]
        means[c] = float(np.nanmean(valid))
        stds[c]  = float(np.nanstd(valid))
        if stds[c] < 1e-6:
            stds[c] = 1.0   # avoid division by zero for constant channels

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path,
             channel_names=np.array(channel_names),
             means=means.astype(np.float32),
             stds=stds.astype(np.float32))
    print(f"[norm] Stats saved to {save_path}")
    for i, ch in enumerate(channel_names):
        print(f"  {ch:15s}  µ={means[i]:.4f}  σ={stds[i]:.4f}")
    return means.astype(np.float32), stds.astype(np.float32)


def normalize(tensor: np.ndarray,
              means: np.ndarray,
              stds: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalisation.
    tensor: (..., C)
    Returns: (..., C) float32 with NaN preserved.
    """
    return ((tensor - means) / stds).astype(np.float32)


def denormalize(tensor: np.ndarray,
                means: np.ndarray,
                stds: np.ndarray) -> np.ndarray:
    """Invert z-score normalisation."""
    return (tensor * stds + means).astype(np.float32)


def normalize_target(target: np.ndarray,
                     mean: float, std: float) -> np.ndarray:
    """Normalize the depth-profile target tensor (same z-score scheme)."""
    return ((target - mean) / std).astype(np.float32)


def denormalize_target(target: np.ndarray,
                       mean: float, std: float) -> np.ndarray:
    return (target * std + mean).astype(np.float32)
