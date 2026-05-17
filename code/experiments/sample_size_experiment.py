"""Sample-size sensitivity experiment."""

from __future__ import annotations

from pathlib import Path

from src.data import generate_synthetic_data
from src.metrics import kendall_tau_distance, pairwise_accuracy, ranking_loss, top_k_overlap
from src.model import LinearRanker
from src.pairs import generate_preference_pairs
from src.train import train_ranker
from src.utils import ExperimentResults, save_results_csv
from src.visualize import plot_sample_size_experiment


def run_sample_size_experiment(
    sample_sizes=None,
    n_features: int = 2,
    noise_std: float = 0.1,
    epochs: int = 200,
    lr: float = 0.05,
    reg: float = 1e-3,
    seed: int = 42,
) -> ExperimentResults:
    if sample_sizes is None:
        sample_sizes = [20, 50, 100, 200, 500]

    rows = []
    for n_samples in sample_sizes:
        X, true_scores, _ = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            noise_std=noise_std,
            seed=seed,
        )
        pairs, labels = generate_preference_pairs(true_scores)
        model = LinearRanker(n_features=n_features, seed=seed)
        train_ranker(model, X, pairs, labels, lr=lr, epochs=epochs, reg=reg, verbose=False)

        pred_scores = model.predict(X)
        rows.append(
            {
                "n_samples": n_samples,
                "noise": noise_std,
                "n_pairs": len(pairs),
                "pairwise_accuracy": pairwise_accuracy(pred_scores, pairs, labels),
                "ranking_loss": ranking_loss(pred_scores, pairs, labels),
                "kendall_tau_distance": kendall_tau_distance(true_scores, pred_scores),
                "top_10_overlap": top_k_overlap(true_scores, pred_scores, k=10),
            }
        )

    results_df = ExperimentResults(rows)
    save_results_csv(rows, Path("results") / "tables" / "sample_size_results.csv")
    plot_sample_size_experiment(
        results_df, Path("results") / "figures" / "sample_size_experiment.png"
    )
    return results_df
