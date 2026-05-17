"""Linear scoring model for pairwise ranking."""

from __future__ import annotations

import numpy as np


class LinearRanker:
    """Linear ranker h_w(x) = w^T x."""

    def __init__(self, n_features: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(loc=0.0, scale=0.01, size=n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.w

    def score_pair(self, X: np.ndarray, i: int, j: int) -> float:
        return float(self.predict(X[[i]])[0] - self.predict(X[[j]])[0])

    def update(self, grad: np.ndarray, lr: float) -> None:
        self.w = self.w - lr * grad

