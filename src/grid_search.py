"""Grid search over the narrowed ranges produced by the Bayesian optimization.

An exhaustive cartesian product over ~10 hyperparameters explodes quickly
(even with only 3 points per hyperparameter it reaches 3^10 ~= 59000 combinations).
Therefore the full grid is, if necessary, randomly (seeded) subsampled down
to `max_combinations`, while still keeping only points on the grid (this is
not continuous sampling as in the Bayesian step).
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import HyperparamRange
from .models import build_model, count_parameters
from .preprocessing import PreprocessedData
from .utils.io import ResultsPaths, save_json, save_table
from .utils.logging_utils import get_logger


def _grid_values_for_range(rng: HyperparamRange, resolution: int) -> list[Any]:
    if rng.kind == "categorical":
        return list(rng.choices)
    if rng.kind == "int":
        points = np.linspace(rng.low, rng.high, resolution)
        return sorted({int(round(p)) for p in points})
    points = np.geomspace(max(rng.low, 1e-12), rng.high, resolution) if rng.log else np.linspace(rng.low, rng.high, resolution)
    return sorted({float(p) for p in points})


def build_grid(
    narrow_ranges: dict[str, HyperparamRange], resolution: int, max_combinations: int, seed: int = 0
) -> list[dict[str, Any]]:
    names = list(narrow_ranges.keys())
    value_lists = [_grid_values_for_range(narrow_ranges[name], resolution) for name in names]

    full_grid = [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]

    if not full_grid:
        raise RuntimeError("The built grid is empty: check the input narrowed ranges")

    if len(full_grid) > max_combinations:
        rng_np = np.random.default_rng(seed)
        idx = rng_np.choice(len(full_grid), size=max_combinations, replace=False)
        return [full_grid[i] for i in sorted(idx)]
    return full_grid


def run_grid_search(
    arch_name: str,
    data: PreprocessedData,
    narrow_ranges: dict[str, HyperparamRange],
    results_paths: ResultsPaths,
    epochs: int,
    resolution: int,
    max_combinations: int,
    seed: int = 0,
) -> dict[str, Any]:
    logger = get_logger("grid_search", results_paths.logs / "grid_search.log")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selected for the grid search: %s", device)

        input_len = data.train.shape[1]
        n_features = data.train.shape[2]

        grid = build_grid(narrow_ranges, resolution, max_combinations, seed)
        logger.info("Grid search: %d combinations to evaluate", len(grid))

        val_tensor = torch.from_numpy(data.val).float().to(device)
        loss_fn = nn.MSELoss()

        records: list[dict[str, Any]] = []
        best_hp: dict[str, Any] | None = None
        best_val_mse = float("inf")
        start_time = time.time()

        for i, hp in enumerate(grid):
            try:
                torch.manual_seed(seed + i)
                model = build_model(arch_name, hp, input_len, n_features).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

                train_loader = DataLoader(
                    TensorDataset(torch.from_numpy(data.train).float()), batch_size=hp["batch_size"], shuffle=True
                )

                combo_start = time.time()
                model.train()
                for _ in range(epochs):
                    for (batch,) in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        recon = model(batch)
                        loss = loss_fn(recon, batch)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_mse = loss_fn(model(val_tensor), val_tensor).item()
                combo_elapsed = time.time() - combo_start

                if not np.isfinite(val_mse):
                    raise ValueError(f"val_mse is not finite: {val_mse}")

                records.append({**hp, "val_mse": val_mse, "n_parameters": count_parameters(model), "seconds": combo_elapsed})

                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_hp = hp
                    logger.info("New best at combination %d/%d: val_mse=%.6g", i + 1, len(grid), val_mse)
            except (ValueError, RuntimeError) as exc:
                logger.warning("Combination %d/%d discarded: %s", i + 1, len(grid), exc)
            except Exception:
                logger.exception("Combination %d/%d failed due to an unexpected error", i + 1, len(grid))
            finally:
                logger.info("Grid search progress: %d/%d combinations done", i + 1, len(grid))

        elapsed = time.time() - start_time

        if best_hp is None:
            raise RuntimeError("No grid search combination produced a valid result")

        import pandas as pd

        save_table(pd.DataFrame(records), results_paths.grid_search / "grid_results", formats=("csv",))
        save_json(best_hp, results_paths.grid_search / "best_hyperparams.json")
        save_json(
            {
                "n_combinations_evaluated": len(grid),
                "n_combinations_successful": len(records),
                "elapsed_seconds": elapsed,
                "mean_seconds_per_combination": elapsed / len(grid) if grid else None,
            },
            results_paths.grid_search / "execution_times.json",
        )

        logger.info("Grid search completed. Best hyperparameters: %s (val_mse=%.6g)", best_hp, best_val_mse)
        return best_hp
    except Exception:
        logger.exception("Grid search failed for architecture '%s'", arch_name)
        raise
