"""Architectures, hyperparameter search space, training defaults, metrics and damage bins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LatentMode = Literal["vector", "sequence"]


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    n_conv_layers: int  # 1 for the first two families, 2 for the third
    use_activation: bool  # ReLU after each convolutional layer
    latent_mode: LatentMode  # "vector": single-vector bottleneck; "sequence": reduced-sequence bottleneck


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "conv_lstm_vec": ArchitectureSpec("conv_lstm_vec", n_conv_layers=1, use_activation=False, latent_mode="vector"),
    "conv_lstm_seq": ArchitectureSpec("conv_lstm_seq", n_conv_layers=1, use_activation=False, latent_mode="sequence"),
    "conv_relu_lstm_vec": ArchitectureSpec("conv_relu_lstm_vec", n_conv_layers=1, use_activation=True, latent_mode="vector"),
    "conv_relu_lstm_seq": ArchitectureSpec("conv_relu_lstm_seq", n_conv_layers=1, use_activation=True, latent_mode="sequence"),
    "conv2_relu_lstm_vec": ArchitectureSpec("conv2_relu_lstm_vec", n_conv_layers=2, use_activation=True, latent_mode="vector"),
    "conv2_relu_lstm_seq": ArchitectureSpec("conv2_relu_lstm_seq", n_conv_layers=2, use_activation=True, latent_mode="sequence"),
}


@dataclass(frozen=True)
class HyperparamRange:
    kind: Literal["categorical", "int", "float"]
    choices: list | None = None
    low: float | None = None
    high: float | None = None
    log: bool = False


def get_search_space(arch_name: str) -> dict[str, HyperparamRange]:
    """Hyperparameter search space (shared across all architectures).

    NOTE: 'padding' is not included here because its range depends on the
    kernel_size sampled in the same trial (0 <= padding <= kernel_size // 2);
    it is sampled directly in bayesian_optimizer.build_objective after
    kernel_size has been sampled.
    """
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURES)}")

    return {
        "n_filters": HyperparamRange("categorical", choices=[8, 16, 32, 64, 128]),
        "kernel_size": HyperparamRange("categorical", choices=[3, 5, 7, 9, 11]),
        "stride": HyperparamRange("categorical", choices=[1, 2, 4, 8]),
        "hidden_units": HyperparamRange("categorical", choices=[16, 32, 64, 128, 256]),
        "latent_dim": HyperparamRange("categorical", choices=[4, 8, 16, 32, 64]),
        "dropout": HyperparamRange("float", low=0.0, high=0.5),
        "weight_decay": HyperparamRange("float", low=1e-6, high=1e-2, log=True),
        "learning_rate": HyperparamRange("float", low=1e-4, high=1e-2, log=True),
        "batch_size": HyperparamRange("categorical", choices=[8, 16, 32, 64]),
    }


# Initial values: the user will fine-tune these manually later.
TRAINING_DEFAULTS: dict[str, int | str] = {
    "hpo_epochs": 20,
    "grid_epochs": 20,
    "final_epochs": 50,
    "optimizer": "adam",
}

# Optimization orchestration defaults (overridable from the CLI in run_experiment).
OPTIMIZATION_DEFAULTS: dict[str, int | float] = {
    "n_hpo_trials": 50,
    "top_n": 10,
    "parsimony_tolerance": 0.05,  # relative tolerance on val_mse for parsimonious selection
    "grid_resolution": 3,  # points per hyperparameter in the narrowed grid search
    "grid_max_combinations": 200,  # cap on the cartesian product, otherwise it explodes (e.g. 3^10)
    "seed": 0,
}

DAMAGE_BIN_EDGES: np.ndarray = np.round(np.arange(0.0, 1.01, 0.1), 2)  # 10 bins: [0-0.1) ... [0.9-1.0]

ERROR_METRICS: list[str] = [
    "mse",
    "mae",
    "rmse",
    "mse_per_feature",
    "mae_per_feature",
    "rmse_per_feature",
    "pearson",
    "kendall_tau_b",
]
