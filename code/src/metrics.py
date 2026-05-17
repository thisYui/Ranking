"""Ranking quality metrics."""

from __future__ import annotations

import numpy as np


def _predicted_pair_labels(pred_scores: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    diffs = pred_scores[pairs[:, 0]] - pred_scores[pairs[:, 1]]
    return np.where(diffs > 0.0, 1, -1)


def pairwise_accuracy(
    pred_scores: np.ndarray, pairs: np.ndarray, labels: np.ndarray
) -> float:
    """Fraction of preference pairs ordered correctly."""
    if len(pairs) == 0:
        return 0.0
    pred_labels = _predicted_pair_labels(np.asarray(pred_scores), np.asarray(pairs))
    return float(np.mean(pred_labels == labels))


def ranking_loss(pred_scores: np.ndarray, pairs: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of preference pairs ordered incorrectly."""
    return 1.0 - pairwise_accuracy(pred_scores, pairs, labels)


def kendall_tau_distance(true_scores: np.ndarray, pred_scores: np.ndarray) -> float:
    """Normalized pairwise disagreement between true and predicted rankings."""
    true_scores = np.asarray(true_scores)
    pred_scores = np.asarray(pred_scores)
    n_samples = len(true_scores)
    total = n_samples * (n_samples - 1) // 2
    if total == 0:
        return 0.0

    disagreements = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            true_order = true_scores[i] > true_scores[j]
            pred_order = pred_scores[i] > pred_scores[j]
            if true_order != pred_order:
                disagreements += 1
    return disagreements / total


def top_k_overlap(true_scores: np.ndarray, pred_scores: np.ndarray, k: int = 10) -> float:
    """Overlap between true and predicted top-k sets."""
    n_samples = len(true_scores)
    if n_samples == 0:
        return 0.0
    k = min(k, n_samples)
    true_top = set(np.argsort(true_scores)[-k:])
    pred_top = set(np.argsort(pred_scores)[-k:])
    return len(true_top.intersection(pred_top)) / k

