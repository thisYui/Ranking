"""Plotting helpers for ranking experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _prepare_path(save_path: str | Path) -> Path:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    return save_path


def plot_loss_curve(history: dict, save_path: str) -> None:
    save_path = _prepare_path(save_path)
    plt.figure(figsize=(7, 4))
    plt.plot(history["loss"], linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Pairwise hinge loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ranking_comparison(true_scores, pred_scores, save_path: str) -> None:
    save_path = _prepare_path(save_path)
    true_order = np.argsort(-np.asarray(true_scores))
    pred_order = np.argsort(-np.asarray(pred_scores))
    x = np.arange(len(true_order))

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, true_order, label="True ranking", linewidth=2)
    plt.plot(x, pred_order, label="Predicted ranking", linewidth=2, linestyle="--")
    plt.title("True Ranking vs Predicted Ranking")
    plt.xlabel("Rank position")
    plt.ylabel("Item index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_noise_experiment(results_df, save_path: str) -> None:
    save_path = _prepare_path(save_path)
    plt.figure(figsize=(7, 4.5))
    plt.plot(results_df["noise"], results_df["pairwise_accuracy"], marker="o", label="Pairwise accuracy")
    plt.plot(results_df["noise"], results_df["ranking_loss"], marker="s", label="Ranking loss")
    plt.plot(results_df["noise"], results_df["kendall_tau_distance"], marker="^", label="Kendall tau distance")
    plt.title("Effect of Noise on Ranking Quality")
    plt.xlabel("Noise standard deviation")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sample_size_experiment(results_df, save_path: str) -> None:
    save_path = _prepare_path(save_path)
    plt.figure(figsize=(7, 4.5))
    plt.plot(results_df["n_samples"], results_df["pairwise_accuracy"], marker="o", label="Pairwise accuracy")
    plt.plot(results_df["n_samples"], results_df["ranking_loss"], marker="s", label="Ranking loss")
    plt.plot(results_df["n_samples"], results_df["kendall_tau_distance"], marker="^", label="Kendall tau distance")
    plt.title("Effect of Sample Size on Ranking Quality")
    plt.xlabel("Number of samples")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

