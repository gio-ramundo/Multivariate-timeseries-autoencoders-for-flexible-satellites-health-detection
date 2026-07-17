"""Windowing/stitching utilities for chunked training and inference on long time series.

Chunking is only about how a fixed-length series is split into overlapping
windows (for training) and how per-window model outputs are merged back into a
full-length series (for evaluation). It knows nothing about models, HPO or I/O:
callers feed it plain (n_instances, series_len, n_features) arrays.
"""

from __future__ import annotations

import numpy as np
from .config import CHUNK_OVERLAP_RATIO


def compute_chunk_starts(series_len: int, chunk_len: int, overlap_ratio: float = CHUNK_OVERLAP_RATIO) -> list[int]:
    """Start indices of sliding windows of length `chunk_len` over a series of
    length `series_len`, consecutive windows overlapping by `overlap_ratio` of the
    chunk length. If the regular stride doesn't reach the last timestep, one extra
    window anchored to the end (start = series_len - chunk_len) is appended, so the
    whole series is covered without any padding.
    """
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}")
    if chunk_len > series_len:
        raise ValueError(f"chunk_len ({chunk_len}) cannot exceed the series length ({series_len})")

    stride = max(1, round(chunk_len * (1 - overlap_ratio)))
    starts = list(range(0, series_len - chunk_len + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] + chunk_len < series_len:
        starts.append(series_len - chunk_len)
    return starts


def chunk_array(
    data: np.ndarray, chunk_len: int, overlap_ratio: float = CHUNK_OVERLAP_RATIO
) -> tuple[np.ndarray, list[int]]:
    """Split `data`, shape (n_instances, series_len, n_features), into overlapping
    windows along the time axis. Returns the stacked windows, shape
    (n_windows * n_instances, chunk_len, n_features), grouped window-major (all
    instances for window 0, then all instances for window 1, ...), and the list of
    window start indices used (see :func:`compute_chunk_starts`).
    """
    n_instances, series_len, n_features = data.shape
    starts = compute_chunk_starts(series_len, chunk_len, overlap_ratio)
    chunks = np.stack([data[:, s : s + chunk_len, :] for s in starts], axis=0)  # (n_windows, n_instances, chunk_len, F)
    return chunks.reshape(len(starts) * n_instances, chunk_len, n_features), starts


def stitch_chunks(chunk_preds: np.ndarray, starts: list[int], n_instances: int, series_len: int) -> np.ndarray:
    """Inverse of :func:`chunk_array`: reassembles window-major stacked chunk
    predictions, shape (n_windows * n_instances, chunk_len, n_features), into
    full-length series, shape (n_instances, series_len, n_features). Steps covered
    by more than one window (the 20% overlap, plus whatever overlap the
    end-anchored extra window introduces) are averaged: predictions are summed into
    an accumulator and each step is divided by the number of windows that covered
    it.
    """
    n_windows = len(starts)
    chunk_len = chunk_preds.shape[1]
    n_features = chunk_preds.shape[2]
    chunk_preds = chunk_preds.reshape(n_windows, n_instances, chunk_len, n_features)

    acc = np.zeros((n_instances, series_len, n_features), dtype=chunk_preds.dtype)
    counts = np.zeros((series_len,), dtype=np.int64)
    for w, s in enumerate(starts):
        acc[:, s : s + chunk_len, :] += chunk_preds[w]
        counts[s : s + chunk_len] += 1

    return acc / counts[None, :, None]
