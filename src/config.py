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


def conv_layer_key(name: str, layer_idx: int) -> str:
    """Per-layer hyperparameter key. layer_idx is 0-indexed, the key suffix is 1-indexed:
    conv_layer_key("n_filters", 0) -> "n_filters_1" (first conv layer)."""
    return f"{name}_{layer_idx + 1}"


# Padding is fixed to 0 (not searched) for every conv/deconv layer: the input
# sequences are long (thousands of timesteps), so the boundary samples a small
# padding would preserve are a negligible fraction of each instance, and fixing
# it removes two dimensions per conv layer from the HPO search space.
CONV_PADDING = 0


def get_search_space(arch_name: str) -> dict[str, HyperparamRange]:
    """Hyperparameter search space for the given architecture.

    n_filters/kernel_size/stride are sampled independently for each
    convolutional layer (keys suffixed by 1-indexed layer number, see
    :func:`conv_layer_key`), so the two conv layers of a 2-layer architecture
    are not forced to share the same values. Padding is not part of the search
    space at all: it is fixed to :data:`CONV_PADDING` for every layer.
    """
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURES)}")

    spec = ARCHITECTURES[arch_name]

    space: dict[str, HyperparamRange] = {
        "hidden_units": HyperparamRange("categorical", choices=[16, 32, 64, 128, 256]),
        "latent_dim": HyperparamRange("categorical", choices=[4, 8, 16, 32, 64]),
        "dropout": HyperparamRange("float", low=0.0, high=0.5),
        "weight_decay": HyperparamRange("float", low=1e-6, high=1e-2, log=True),
        "learning_rate": HyperparamRange("float", low=1e-4, high=1e-2, log=True),
        "batch_size": HyperparamRange("categorical", choices=[8, 16, 32, 64]),
    }
    for layer_idx in range(spec.n_conv_layers):
        space[conv_layer_key("n_filters", layer_idx)] = HyperparamRange("categorical", choices=[8, 16, 32, 64, 128])
        space[conv_layer_key("kernel_size", layer_idx)] = HyperparamRange("categorical", choices=[3, 5, 7, 9, 11])
        space[conv_layer_key("stride", layer_idx)] = HyperparamRange("categorical", choices=[1, 2, 4, 8])
    return space


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
    "n_jobs_hpo": 1,  # parallel Optuna trials (threads); >1 reduces exact seed-reproducibility
    "n_jobs_gs": 1,  # parallel grid search combinations (threads); >1 reduces exact seed-reproducibility
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
