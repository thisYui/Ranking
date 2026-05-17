"""Synthetic data generation for ranking experiments."""

from __future__ import annotations

import numpy as np


def generate_synthetic_data(
    n_samples: int = 100,
    n_features: int = 2,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a linear ranking problem with Gaussian feature noise."""
    feature_rng = np.random.default_rng(seed)
    weight_rng = np.random.default_rng(seed + 1009)
    noise_rng = np.random.default_rng(seed + 2027)

    X = feature_rng.standard_normal((n_samples, n_features))
    true_w = weight_rng.standard_normal(n_features)
    clean_scores = X @ true_w
    noise = noise_rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    true_scores = clean_scores + noise
    return X, true_scores, true_w
