"""Run all supplementary ranking experiments."""

from __future__ import annotations

from experiments.noise_experiment import run_noise_experiment
from experiments.sample_size_experiment import run_sample_size_experiment
from src.utils import ensure_dir, set_seed


def main() -> None:
    set_seed(42)
    ensure_dir("results/figures")
    ensure_dir("results/tables")

    print("Running noise experiment...")
    noise_results = run_noise_experiment()
    print(noise_results.to_string(index=False))

    print("\nRunning sample size experiment...")
    sample_size_results = run_sample_size_experiment()
    print(sample_size_results.to_string(index=False))

    print("\nSaved experiment outputs under results/figures/ and results/tables/.")


if __name__ == "__main__":
    main()

