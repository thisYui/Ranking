"""Training loop for the pairwise linear ranker."""

from __future__ import annotations

from .loss import pairwise_hinge_loss_and_grad
from .metrics import pairwise_accuracy, ranking_loss


def train_ranker(
    model,
    X,
    pairs,
    labels,
    lr: float = 0.05,
    epochs: int = 200,
    reg: float = 1e-3,
    verbose: bool = True,
) -> dict:
    """Optimize pairwise hinge loss with full-batch gradient descent."""
    history = {"loss": [], "pairwise_accuracy": [], "ranking_loss": []}

    for epoch in range(1, epochs + 1):
        loss, grad = pairwise_hinge_loss_and_grad(model, X, pairs, labels, reg=reg)
        model.update(grad, lr=lr)

        pred_scores = model.predict(X)
        history["loss"].append(loss)
        history["pairwise_accuracy"].append(pairwise_accuracy(pred_scores, pairs, labels))
        history["ranking_loss"].append(ranking_loss(pred_scores, pairs, labels))

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            acc = history["pairwise_accuracy"][-1]
            print(f"epoch={epoch:03d} loss={loss:.4f} pairwise_accuracy={acc:.4f}")

    return history

