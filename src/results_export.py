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


def plot_error_vs_damage_parameter(errors_df: pd.DataFrame, out_path: Path, metric: str = "mse") -> None:
    damage_types = [g for g in DAMAGE_ORDER if g in errors_df["dataset"].unique()]

    fig, ax = plt.subplots(figsize=(6, 4))
    for dt in damage_types:
        sub = errors_df[errors_df["dataset"] == dt]
        ax.scatter(sub["damage_parameter"], sub[metric], label=dt, alpha=0.6)
        if len(sub) > 1:
            coeffs = np.polyfit(sub["damage_parameter"], sub[metric], 1)
            xs = np.linspace(0, 1, 50)
            ax.plot(xs, np.polyval(coeffs, xs), linestyle="--")

    ax.set_xlabel("damage parameter")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs damage parameter")
    ax.legend()
    save_figure(fig, out_path)
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


def plot_sample_reconstructions(x: np.ndarray, x_hat: np.ndarray, out_path: Path, n_samples: int = 5, feature_idx: int = 0) -> None:
    n = min(n_samples, x.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for i in range(n):
        axes[i].plot(x[i, :, feature_idx], label="original")
        axes[i].plot(x_hat[i, :, feature_idx], label="reconstructed", linestyle="--")
        axes[i].set_ylabel(f"instance {i}")

    axes[0].legend()
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"Sample reconstruction (feature index {feature_idx})")
    save_figure(fig, out_path)
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


def export_all_results(data: PreprocessedData, train_test_results: dict[str, Any], results_paths: ResultsPaths) -> None:
    logger = get_logger("results_export", results_paths.logs / "results_export.log")

    try:
        curve = train_test_results["training_curve"]
        errors_df = train_test_results["errors"]
        predictions = train_test_results["predictions"]

        plot_training_curves(curve, results_paths.figures / "training_curve.pdf")

        for metric in METRICS:
            plot_error_boxplot(errors_df, results_paths.figures / f"error_boxplot_{metric}.pdf", metric=metric)
            plot_error_vs_damage_parameter(errors_df, results_paths.figures / f"error_vs_damage_{metric}.pdf", metric=metric)
            table = build_damage_range_table(errors_df, DAMAGE_BIN_EDGES, metric=metric)
            save_table(table, results_paths.tables / f"damage_range_table_{metric}", formats=("csv", "xlsx"))

        aucs = plot_roc_curve(errors_df, results_paths.figures / "roc_curve.pdf", metric="mse")
        save_table(pd.DataFrame({"auc": aucs}), results_paths.tables / "roc_auc", formats=("csv",))

        plot_sample_reconstructions(data.test, predictions["healthy"], results_paths.figures / "sample_reconstructions_healthy.pdf")
        for damage_type in data.damage:
            plot_sample_reconstructions(
                data.damage[damage_type], predictions[damage_type], results_paths.figures / f"sample_reconstructions_{damage_type}.pdf"
            )

        logger.info("Results export completed in %s", results_paths.figures)
    except Exception:
        logger.exception("Results export failed")
        raise
