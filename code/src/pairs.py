"""Preference-pair construction utilities."""

from __future__ import annotations

import numpy as np


def generate_preference_pairs(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create all pairwise preferences from descending target scores."""
    scores = np.asarray(scores)
    n_samples = scores.shape[0]
    pairs: list[tuple[int, int]] = []
    labels: list[int] = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairs.append((i, j))
            labels.append(1 if scores[i] > scores[j] else -1)

    return np.asarray(pairs, dtype=int), np.asarray(labels, dtype=int)

