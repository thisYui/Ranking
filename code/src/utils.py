"""General project utilities."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_results_csv(rows: list[dict], save_path: str | Path) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    if not rows:
        save_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


class ExperimentResults:
    """Small table-like result object used to avoid heavyweight dependencies."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def __getitem__(self, column: str) -> list:
        return [row[column] for row in self.rows]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def to_string(self, index: bool = False) -> str:
        if not self.rows:
            return ""

        formatted_rows = [
            {key: _format_cell(value) for key, value in row.items()} for row in self.rows
        ]
        widths = {
            key: max(len(key), *(len(row[key]) for row in formatted_rows))
            for key in self.columns
        }
        header = " ".join(key.rjust(widths[key]) for key in self.columns)
        lines = [header]
        for row in formatted_rows:
            lines.append(" ".join(row[key].rjust(widths[key]) for key in self.columns))
        return "\n".join(lines)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
