# Foundations of Machine Learning - Ranking

This repository contains the report workspace and code implementation for the Ranking topic.

## Submission Metadata

- Group ID: TODO
- Members and student IDs: TODO
- Book chapter/topic: Foundations of Machine Learning - Ranking
- Main focus: Pairwise learning to rank

## Completed Code Work

The `code/` folder implements a reproducible from-scratch NumPy pipeline:

```text
Synthetic data -> preference pairs -> linear ranker -> pairwise hinge loss
-> gradient descent -> ranking metrics -> CSV tables and figures
```

The implementation does not use scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, PyTorch, or JAX for the main ranking algorithm.

## Run From Repository Root

```bash
pip install -r requirements.txt
python run_main.py
python run_experiments.py
```

## Run From `code/`

```bash
cd code
pip install -r requirements.txt
python run_main.py
python run_experiments.py
```

## Generated Outputs

After running the scripts, the following files are generated:

```text
code/results/figures/loss_curve.png
code/results/figures/ranking_comparison.png
code/results/figures/noise_experiment.png
code/results/figures/sample_size_experiment.png
code/results/tables/main_results.csv
code/results/tables/noise_results.csv
code/results/tables/sample_size_results.csv
```

## Notes

Replace the TODO metadata above before packaging the final submission.
