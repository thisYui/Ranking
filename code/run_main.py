"""Run the main pairwise learning-to-rank experiment."""

from __future__ import annotations

from pathlib import Path

from src.data import generate_synthetic_data
from src.metrics import kendall_tau_distance, pairwise_accuracy, ranking_loss, top_k_overlap
from src.model import LinearRanker
from src.pairs import generate_preference_pairs
from src.train import train_ranker
from src.utils import ensure_dir, save_results_csv, set_seed
from src.visualize import plot_loss_curve, plot_ranking_comparison


def main() -> None:
    n_samples = 100
    n_features = 2
    noise_std = 0.1
    epochs = 200
    learning_rate = 0.05
    regularization = 1e-3
    seed = 42

    set_seed(seed)
    ensure_dir(Path("results") / "figures")
    ensure_dir(Path("results") / "tables")

    X, true_scores, true_w = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std=noise_std,
        seed=seed,
    )
    pairs, labels = generate_preference_pairs(true_scores)

    model = LinearRanker(n_features=n_features, seed=seed)
    history = train_ranker(
        model,
        X,
        pairs,
        labels,
        lr=learning_rate,
        epochs=epochs,
        reg=regularization,
        verbose=True,
    )

    pred_scores = model.predict(X)
    row = {
        "n_samples": n_samples,
        "n_features": n_features,
        "noise": noise_std,
        "n_pairs": len(pairs),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "pairwise_accuracy": pairwise_accuracy(pred_scores, pairs, labels),
        "ranking_loss": ranking_loss(pred_scores, pairs, labels),
        "kendall_tau_distance": kendall_tau_distance(true_scores, pred_scores),
        "top_10_overlap": top_k_overlap(true_scores, pred_scores, k=10),
        "true_w": " ".join(f"{value:.6f}" for value in true_w),
        "learned_w": " ".join(f"{value:.6f}" for value in model.w),
        "final_training_loss": history["loss"][-1],
    }

    save_results_csv([row], Path("results") / "tables" / "main_results.csv")
    plot_loss_curve(history, Path("results") / "figures" / "loss_curve.png")
    plot_ranking_comparison(
        true_scores, pred_scores, Path("results") / "figures" / "ranking_comparison.png"
    )

    print("\nFinal metrics")
    for key in ["pairwise_accuracy", "ranking_loss", "kendall_tau_distance", "top_10_overlap"]:
        print(f"{key}: {row[key]:.4f}")
    print("Saved results under results/figures/ and results/tables/.")


if __name__ == "__main__":
    main()

