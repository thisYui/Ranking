"""Pairwise hinge loss and gradient implemented from scratch."""

from __future__ import annotations

import numpy as np


def pairwise_hinge_loss_and_grad(
    model,
    X: np.ndarray,
    pairs: np.ndarray,
    labels: np.ndarray,
    reg: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return mean pairwise hinge loss and gradient with L2 regularization."""
    X = np.asarray(X)
    pairs = np.asarray(pairs, dtype=int)
    labels = np.asarray(labels)

    if len(pairs) == 0:
        loss = 0.5 * reg * float(model.w @ model.w)
        grad = reg * model.w
        return loss, grad

    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    margins = labels * (diffs @ model.w)
    violations = margins < 1.0

    hinge_losses = np.maximum(0.0, 1.0 - margins)
    loss = float(np.mean(hinge_losses))

    grad = np.zeros_like(model.w)
    if np.any(violations):
        grad = np.mean((-labels[violations, None]) * diffs[violations], axis=0)

    loss += 0.5 * reg * float(model.w @ model.w)
    grad += reg * model.w
    return loss, grad

