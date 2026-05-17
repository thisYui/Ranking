# Pairwise Learning to Rank

This folder contains a reproducible Python implementation for the Ranking topic in Foundations of Machine Learning. The main algorithm is a from-scratch NumPy implementation of pairwise learning to rank with a linear scoring model and pairwise hinge loss.

## Goal

The project generates synthetic ranking data, converts true scores into preference pairs, trains a linear ranker, evaluates ranking quality, and saves figures and CSV tables for the report.

## Folder Structure

```text
code/
├── run_main.py
├── run_experiments.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── pairs.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── metrics.py
│   ├── visualize.py
│   └── utils.py
├── experiments/
│   ├── noise_experiment.py
│   └── sample_size_experiment.py
├── results/
│   ├── figures/
│   └── tables/
└── tests/
    └── test_metrics.py
```

## Installation

Run these commands from the `code/` directory:

```bash
pip install -r requirements.txt
```

## Main Experiment

```bash
python run_main.py
```

This trains the pairwise linear ranker with the default setting:

```text
n_samples = 100
n_features = 2
noise_std = 0.1
epochs = 200
learning_rate = 0.05
regularization = 1e-3
seed = 42
```

Generated outputs:

```text
results/figures/loss_curve.png
results/figures/ranking_comparison.png
results/tables/main_results.csv
```

## All Experiments

```bash
python run_experiments.py
```

Generated outputs:

```text
results/figures/noise_experiment.png
results/figures/sample_size_experiment.png
results/tables/noise_results.csv
results/tables/sample_size_results.csv
```

## Metrics

The implementation reports:

- Pairwise accuracy
- Ranking loss
- Kendall tau distance
- Top-10 overlap

## Notes

The ranking algorithm is implemented from scratch with NumPy. The project does not use scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, PyTorch, or JAX for the main ranking method.

