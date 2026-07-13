"""Export of plots (PDF) and tables (CSV/XLSX) from the train_test results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .config import DAMAGE_BIN_EDGES
from .preprocessing import PreprocessedData
from .utils.io import ResultsPaths, save_figure, save_json, save_table
from .utils.logging_utils import get_logger

DAMAGE_ORDER = ("stiffness", "gyroscope", "torque")
METRICS = ("mse", "mae", "rmse")


def plot_training_curves(curve: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["epoch"], curve["train_loss"], label="train")
    ax.plot(curve["epoch"], curve["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training curves")
    ax.legend()
    save_figure(fig, out_path)
    plt.close(fig)


def plot_error_boxplot(errors_df: pd.DataFrame, out_path: Path, metric: str = "mse") -> None:
    groups = [g for g in ("healthy", *DAMAGE_ORDER) if g in errors_df["dataset"].unique()]
    values = [errors_df.loc[errors_df["dataset"] == g, metric].dropna().to_numpy() for g in groups]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=groups)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Error distribution ({metric.upper()}) by damage type")
    save_figure(fig, out_path)
    plt.close(fig)


def _draw_error_vs_damage(
    ax: plt.Axes, sub: pd.DataFrame, y_col: str, bin_edges: np.ndarray, bin_labels: list[str], bin_midpoints: np.ndarray
) -> None:
    """Draw, onto an existing ax: scatter of y_col vs damage_parameter, a linear
    regression fit, and a polyline connecting each bin's midpoint (x) to the mean
    of y_col within that bin (y). Shared by the aggregate and per-feature plots."""
    ax.scatter(sub["damage_parameter"], sub[y_col], alpha=0.6, label="instances")

    if len(sub) > 1:
        coeffs = np.polyfit(sub["damage_parameter"], sub[y_col], 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, np.polyval(coeffs, xs), linestyle="--", label="linear fit")

    bins = pd.cut(sub["damage_parameter"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    bin_means_map = sub.groupby(bins, observed=True)[y_col].mean()
    bin_means = [bin_means_map.get(label, np.nan) for label in bin_labels]
    ax.plot(bin_midpoints, bin_means, marker="o", linestyle="-", label="binned mean")


def plot_error_vs_damage_parameter(
    errors_df: pd.DataFrame, out_path: Path, metric: str = "mse", bin_edges: np.ndarray = DAMAGE_BIN_EDGES
) -> None:
    """One figure per damage type (all types overlaid would be unreadable): scatter of
    instance-level error vs damage parameter, a linear regression fit, and a polyline
    connecting each bin's midpoint (x) to the mean error within that bin (y)."""
    damage_types = [g for g in DAMAGE_ORDER if g in errors_df["dataset"].unique()]
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    for dt in damage_types:
        sub = errors_df[errors_df["dataset"] == dt]

        fig, ax = plt.subplots(figsize=(6, 4))
        _draw_error_vs_damage(ax, sub, metric, bin_edges, bin_labels, bin_midpoints)

        ax.set_xlabel("damage parameter")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs damage parameter ({dt})")
        ax.legend()

        dt_path = out_path.with_name(f"{out_path.stem}_{dt}{out_path.suffix}")
        save_figure(fig, dt_path)
        plt.close(fig)


def plot_error_vs_damage_parameter_per_feature(
    errors_per_feature_df: pd.DataFrame, out_dir: Path, metric: str = "mse", bin_edges: np.ndarray = DAMAGE_BIN_EDGES
) -> None:
    """Same plot as plot_error_vs_damage_parameter, but one figure per (feature,
    damage type), organized as out_dir/<metric>/feature<i>_<damage_type>.pdf."""
    damage_types = [g for g in DAMAGE_ORDER if g in errors_per_feature_df["dataset"].unique()]
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    prefix = f"{metric}_f"
    feature_indices = sorted(
        int(c[len(prefix) :]) for c in errors_per_feature_df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()
    )
    if not feature_indices:
        raise ValueError(f"No '{prefix}<i>' columns found in errors_per_feature_df for metric '{metric}'")

    metric_dir = out_dir / metric
    for feature_idx in feature_indices:
        col = f"{prefix}{feature_idx}"
        for dt in damage_types:
            sub = errors_per_feature_df[errors_per_feature_df["dataset"] == dt]

            fig, ax = plt.subplots(figsize=(6, 4))
            _draw_error_vs_damage(ax, sub, col, bin_edges, bin_labels, bin_midpoints)

            ax.set_xlabel("damage parameter")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs damage parameter (feature {feature_idx}, {dt})")
            ax.legend()

            save_figure(fig, metric_dir / f"feature{feature_idx}_{dt}.pdf")
            plt.close(fig)


def plot_roc_curve(errors_df: pd.DataFrame, out_path: Path, metric: str = "mse") -> dict[str, float]:
    healthy_scores = errors_df.loc[errors_df["dataset"] == "healthy", metric].to_numpy()
    damage_types = [g for g in DAMAGE_ORDER if g in errors_df["dataset"].unique()]

    fig, ax = plt.subplots(figsize=(5, 5))
    aucs: dict[str, float] = {}
    for dt in damage_types:
        damage_scores = errors_df.loc[errors_df["dataset"] == dt, metric].to_numpy()
        y_true = np.concatenate([np.zeros_like(healthy_scores), np.ones_like(damage_scores)])
        y_score = np.concatenate([healthy_scores, damage_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = float(auc(fpr, tpr))
        aucs[dt] = auc_value
        ax.plot(fpr, tpr, label=f"{dt} (AUC={auc_value:.3f})")

    ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC (healthy vs damage, {metric.upper()})")
    ax.legend()
    save_figure(fig, out_path)
    plt.close(fig)
    return aucs


def plot_sample_reconstructions(x: np.ndarray, x_hat: np.ndarray, out_path: Path, n_samples: int = 4, seed: int = 0) -> None:
    """One figure per feature (all features in a single figure would be too many
    plots to read): n_samples instances, randomly sampled, original vs
    reconstructed for that one feature, stacked as subplots."""
    n = min(n_samples, x.shape[0])
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(x.shape[0], size=n, replace=False))
    n_features = x.shape[-1]

    for feature_idx in range(n_features):
        fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
        axes = np.atleast_1d(axes)

        for row, instance_idx in enumerate(sample_idx):
            axes[row].plot(x[instance_idx, :, feature_idx], label="original")
            axes[row].plot(x_hat[instance_idx, :, feature_idx], label="reconstructed", linestyle="--")
            axes[row].set_ylabel(f"instance {instance_idx}")

        axes[0].legend()
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"Sample reconstruction (feature index {feature_idx})")

        feature_path = out_path.with_name(f"{out_path.stem}_feature{feature_idx}{out_path.suffix}")
        save_figure(fig, feature_path)
        plt.close(fig)


def build_damage_range_table(errors_df: pd.DataFrame, bin_edges: np.ndarray, metric: str = "mse") -> pd.DataFrame:
    damage_types = [g for g in DAMAGE_ORDER if g in errors_df["dataset"].unique()]
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}" for i in range(len(bin_edges) - 1)]

    table = pd.DataFrame(index=damage_types, columns=bin_labels, dtype=float)
    for dt in damage_types:
        sub = errors_df[errors_df["dataset"] == dt]
        bins = pd.cut(sub["damage_parameter"], bins=bin_edges, labels=bin_labels, include_lowest=True)
        means = sub.groupby(bins, observed=True)[metric].mean()
        for label in bin_labels:
            table.loc[dt, label] = means.get(label, np.nan)

    return table


def export_all_results(
    data: PreprocessedData, train_test_results: dict[str, Any], results_paths: ResultsPaths, seed: int = 0
) -> None:
    logger = get_logger("results_export", results_paths.logs / "results_export.log")

    try:
        curve = train_test_results["training_curve"]
        errors_df = train_test_results["errors"]
        errors_per_feature_df = train_test_results["errors_per_feature"]
        predictions = train_test_results["predictions"]

        plot_training_curves(curve, results_paths.figures / "training_curve.pdf")

        error_per_feature_dir = results_paths.figures / "error_per_feature_vs_damage_param"

        for metric in METRICS:
            plot_error_boxplot(errors_df, results_paths.figures / f"error_boxplot_{metric}.pdf", metric=metric)
            plot_error_vs_damage_parameter(errors_df, results_paths.figures / f"error_vs_damage_{metric}.pdf", metric=metric)
            plot_error_vs_damage_parameter_per_feature(errors_per_feature_df, error_per_feature_dir, metric=metric)
            table = build_damage_range_table(errors_df, DAMAGE_BIN_EDGES, metric=metric)
            save_table(table, results_paths.tables / f"damage_range_table_{metric}", formats=("csv", "xlsx"))

        aucs = plot_roc_curve(errors_df, results_paths.figures / "roc_curve.pdf", metric="mse")
        save_table(pd.DataFrame({"auc": aucs}), results_paths.tables / "roc_auc", formats=("csv",))

        plot_sample_reconstructions(
            data.test, predictions["healthy"], results_paths.figures / "sample_reconstructions_healthy.pdf", seed=seed
        )
        for damage_type in data.damage:
            plot_sample_reconstructions(
                data.damage[damage_type],
                predictions[damage_type],
                results_paths.figures / f"sample_reconstructions_{damage_type}.pdf",
                seed=seed,
            )

        logger.info("Results export completed in %s", results_paths.figures)
    except Exception:
        logger.exception("Results export failed")
        raise
