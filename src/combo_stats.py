"""Statistics and plots for the multi-damage interaction analysis: how the
reconstruction error responds to stiffness/torque/gyroscope severity, alone
and in combination.

Three complementary views are produced for each error metric (mse/mae/rmse):
  - categorical (LOW/MEDIUM/HIGH), based on which combo file an instance came
    from: overview boxplot, main-effects plot, pairwise interaction plots,
    2-way heatmaps, and a 3-way factorial ANOVA table (closed-form, since the
    design is balanced: 27 combos x 30 instances each);
  - continuous, based on the per-instance damage-parameter values: pairwise
    scatter colored by error, and an OLS regression with interaction terms;
  - detectability: per-combo AUC against the healthy baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, roc_curve

from .combo_data import COMBO_DAMAGE_ORDER, LEVEL_NAMES
from .utils.io import save_figure, save_json, save_table

METRICS: tuple[str, ...] = ("mse", "mae", "rmse")


def _severity_sum(combo: str) -> int:
    return sum({"L": 1, "M": 2, "H": 3}[ch] for ch in combo)


def build_combo_errors_table(
    healthy_total: pd.DataFrame,
    combo_totals: dict[str, pd.DataFrame],
    combo_levels: dict[str, dict[str, int]],
    combo_params: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """One row per instance (healthy + every combo), with dataset/level/param
    columns added alongside the error columns already in `*_total` (mse/mae/rmse)."""
    rows = [
        healthy_total.assign(
            dataset="healthy",
            **{f"{dt}_level": 0 for dt in COMBO_DAMAGE_ORDER},
            **{f"{dt}_param": np.nan for dt in COMBO_DAMAGE_ORDER},
        )
    ]
    for combo, total_df in combo_totals.items():
        levels = combo_levels[combo]
        params = combo_params[combo]
        rows.append(
            total_df.assign(
                dataset=combo,
                **{f"{dt}_level": levels[dt] for dt in COMBO_DAMAGE_ORDER},
                **{f"{dt}_param": params[dt] for dt in COMBO_DAMAGE_ORDER},
            )
        )
    return pd.concat(rows, ignore_index=True)


def _ordered_combos(df: pd.DataFrame) -> list[str]:
    return sorted((c for c in df["dataset"].unique() if c != "healthy"), key=_severity_sum)


def plot_overview_boxplot(df: pd.DataFrame, out_path: Path, metric: str) -> None:
    """Boxplot of `metric`: healthy, then the 27 combos ordered by total
    severity (LOW=1/MEDIUM=2/HIGH=3 summed across the 3 damage types)."""
    order = ["healthy", *_ordered_combos(df)]
    values = [df.loc[df["dataset"] == g, metric].dropna().to_numpy() for g in order]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(values, tick_labels=order)
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} distribution: healthy vs the 27 damage combinations")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_main_effects(df: pd.DataFrame, out_path: Path, metric: str) -> None:
    """One figure, one panel per damage type: mean +/- std of `metric` at
    LOW/MEDIUM/HIGH, marginalized over the other two damage types."""
    combo_df = df[df["dataset"] != "healthy"]

    fig, axes = plt.subplots(1, len(COMBO_DAMAGE_ORDER), figsize=(5 * len(COMBO_DAMAGE_ORDER), 4), sharey=True)
    for ax, dt in zip(axes, COMBO_DAMAGE_ORDER):
        grouped = combo_df.groupby(f"{dt}_level")[metric].agg(["mean", "std"]).reindex([1, 2, 3])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([LEVEL_NAMES[level] for level in (1, 2, 3)])
        ax.set_xlabel(dt)
        ax.set_title(f"Main effect: {dt}")
    axes[0].set_ylabel(metric.upper())
    fig.suptitle(f"Main effects on {metric.upper()}")
    save_figure(fig, out_path)
    plt.close(fig)


def _damage_pairs() -> list[tuple[str, str, str]]:
    """(a, b, marginalized_over) for the 3 unordered pairs of damage types."""
    pairs = []
    for i in range(len(COMBO_DAMAGE_ORDER)):
        for j in range(i + 1, len(COMBO_DAMAGE_ORDER)):
            a, b = COMBO_DAMAGE_ORDER[i], COMBO_DAMAGE_ORDER[j]
            (third,) = [d for d in COMBO_DAMAGE_ORDER if d not in (a, b)]
            pairs.append((a, b, third))
    return pairs


def plot_interaction_2way(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Classic factorial-design interaction plot for each damage pair: x-axis is
    one damage's level, one line per level of the other damage, marginalized
    over the third. Non-parallel or crossing lines indicate an interaction
    effect between the two damages (beyond their individual main effects)."""
    combo_df = df[df["dataset"] != "healthy"]

    for a, b, third in _damage_pairs():
        grouped = combo_df.groupby([f"{a}_level", f"{b}_level"])[metric].mean().unstack(f"{b}_level")
        grouped = grouped.reindex(index=[1, 2, 3], columns=[1, 2, 3])

        fig, ax = plt.subplots(figsize=(6, 5))
        for b_level in (1, 2, 3):
            ax.plot(grouped.index, grouped[b_level], marker="o", label=f"{b}={LEVEL_NAMES[b_level]}")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([LEVEL_NAMES[level] for level in (1, 2, 3)])
        ax.set_xlabel(a)
        ax.set_ylabel(f"mean {metric.upper()}")
        ax.set_title(f"Interaction {a} x {b} (marginalized over {third}) - {metric.upper()}")
        ax.legend(title=b)
        save_figure(fig, out_dir / f"{a}_x_{b}.pdf")
        plt.close(fig)


def plot_heatmap_2way(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """3x3 heatmap of mean `metric` for each damage pair, one figure per level
    of the third (marginalized-out) damage type: richer version of
    plot_interaction_2way, split into small multiples instead of overlaid lines."""
    combo_df = df[df["dataset"] != "healthy"]

    for a, b, third in _damage_pairs():
        for third_level in (1, 2, 3):
            sub = combo_df[combo_df[f"{third}_level"] == third_level]
            pivot = sub.groupby([f"{a}_level", f"{b}_level"])[metric].mean().unstack(f"{b}_level")
            pivot = pivot.reindex(index=[1, 2, 3], columns=[1, 2, 3])
            values = pivot.to_numpy()

            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(values, cmap="viridis", origin="lower")
            ax.set_xticks(range(3))
            ax.set_xticklabels([LEVEL_NAMES[level] for level in (1, 2, 3)])
            ax.set_yticks(range(3))
            ax.set_yticklabels([LEVEL_NAMES[level] for level in (1, 2, 3)])
            ax.set_xlabel(b)
            ax.set_ylabel(a)
            ax.set_title(f"{metric.upper()} mean, {third}={LEVEL_NAMES[third_level]}")
            for i in range(3):
                for j in range(3):
                    if not np.isnan(values[i, j]):
                        ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", color="white")
            fig.colorbar(im, ax=ax)
            save_figure(fig, out_dir / f"{a}_x_{b}" / f"{third}_{LEVEL_NAMES[third_level].lower()}.pdf")
            plt.close(fig)


def compute_three_way_anova(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Closed-form 3-way factorial ANOVA (fixed effects) decomposing the
    variance of `metric` into the 3 main effects, the 3 two-way interactions,
    the one three-way interaction, and the residual.

    Requires a balanced design (every combination of the 3 damage levels has
    the same number of instances), which is what the 27 combo files provide;
    an unbalanced design would need per-cell-weighted (type II/III) sums of
    squares instead, which this closed-form version does not handle.
    """
    combo_df = df[df["dataset"] != "healthy"]
    a_col, b_col, c_col = (f"{dt}_level" for dt in COMBO_DAMAGE_ORDER)

    cell_counts = combo_df.groupby([a_col, b_col, c_col])[metric].count()
    if cell_counts.nunique() != 1:
        raise ValueError(
            "compute_three_way_anova requires a balanced design (equal instances per damage-level "
            f"combination), got cell sizes ranging {cell_counts.min()}-{cell_counts.max()}"
        )
    n = int(cell_counts.iloc[0])
    a_levels = b_levels = c_levels = (1, 2, 3)

    grand_mean = combo_df[metric].mean()
    mean_a = combo_df.groupby(a_col)[metric].mean()
    mean_b = combo_df.groupby(b_col)[metric].mean()
    mean_c = combo_df.groupby(c_col)[metric].mean()
    mean_ab = combo_df.groupby([a_col, b_col])[metric].mean()
    mean_ac = combo_df.groupby([a_col, c_col])[metric].mean()
    mean_bc = combo_df.groupby([b_col, c_col])[metric].mean()
    mean_abc = combo_df.groupby([a_col, b_col, c_col])[metric].mean()

    n_a, n_b, n_c = len(a_levels), len(b_levels), len(c_levels)

    ss_a = n_b * n_c * n * sum((mean_a[i] - grand_mean) ** 2 for i in a_levels)
    ss_b = n_a * n_c * n * sum((mean_b[j] - grand_mean) ** 2 for j in b_levels)
    ss_c = n_a * n_b * n * sum((mean_c[k] - grand_mean) ** 2 for k in c_levels)

    ss_ab = n_c * n * sum(
        (mean_ab[i, j] - mean_a[i] - mean_b[j] + grand_mean) ** 2 for i in a_levels for j in b_levels
    )
    ss_ac = n_b * n * sum(
        (mean_ac[i, k] - mean_a[i] - mean_c[k] + grand_mean) ** 2 for i in a_levels for k in c_levels
    )
    ss_bc = n_a * n * sum(
        (mean_bc[j, k] - mean_b[j] - mean_c[k] + grand_mean) ** 2 for j in b_levels for k in c_levels
    )

    ss_abc = n * sum(
        (
            mean_abc[i, j, k]
            - mean_ab[i, j] - mean_ac[i, k] - mean_bc[j, k]
            + mean_a[i] + mean_b[j] + mean_c[k]
            - grand_mean
        )
        ** 2
        for i in a_levels
        for j in b_levels
        for k in c_levels
    )

    ss_error = sum(
        ((group[metric] - mean_abc[key]) ** 2).sum()
        for key, group in combo_df.groupby([a_col, b_col, c_col])
    )
    ss_total = ((combo_df[metric] - grand_mean) ** 2).sum()

    stiffness_dt, torque_dt, gyro_dt = COMBO_DAMAGE_ORDER
    rows = [
        (stiffness_dt, ss_a, n_a - 1),
        (torque_dt, ss_b, n_b - 1),
        (gyro_dt, ss_c, n_c - 1),
        (f"{stiffness_dt}:{torque_dt}", ss_ab, (n_a - 1) * (n_b - 1)),
        (f"{stiffness_dt}:{gyro_dt}", ss_ac, (n_a - 1) * (n_c - 1)),
        (f"{torque_dt}:{gyro_dt}", ss_bc, (n_b - 1) * (n_c - 1)),
        (f"{stiffness_dt}:{torque_dt}:{gyro_dt}", ss_abc, (n_a - 1) * (n_b - 1) * (n_c - 1)),
    ]
    df_error = n_a * n_b * n_c * (n - 1)

    table = pd.DataFrame(rows, columns=["source", "sum_sq", "df"])
    table["mean_sq"] = table["sum_sq"] / table["df"]
    ms_error = ss_error / df_error
    table["F"] = table["mean_sq"] / ms_error
    table["p_value"] = stats.f.sf(table["F"], table["df"], df_error)
    table["eta_sq_partial"] = table["sum_sq"] / (table["sum_sq"] + ss_error)

    residual_row = pd.DataFrame(
        [{"source": "Residual", "sum_sq": ss_error, "df": df_error, "mean_sq": ms_error, "F": np.nan, "p_value": np.nan, "eta_sq_partial": np.nan}]
    )
    total_row = pd.DataFrame([{"source": "Total", "sum_sq": ss_total, "df": len(combo_df) - 1, "mean_sq": np.nan, "F": np.nan, "p_value": np.nan, "eta_sq_partial": np.nan}])
    return pd.concat([table, residual_row, total_row], ignore_index=True)


def plot_continuous_scatter_2d(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """For each damage pair: scatter of the two continuous damage parameters,
    colored by `metric`, pooling instances across all levels of the third
    (marginalized-out) damage type. Continuous counterpart of the 2-way
    heatmap, using each instance's exact parameter value instead of its L/M/H bin."""
    combo_df = df[df["dataset"] != "healthy"]

    for a, b, third in _damage_pairs():
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(combo_df[f"{a}_param"], combo_df[f"{b}_param"], c=combo_df[metric], cmap="viridis", s=15, alpha=0.8)
        ax.set_xlabel(f"{a} parameter")
        ax.set_ylabel(f"{b} parameter")
        ax.set_title(f"{metric.upper()} vs {a}/{b} parameter (all {third} levels pooled)")
        fig.colorbar(sc, ax=ax, label=metric.upper())
        save_figure(fig, out_dir / f"{a}_x_{b}.pdf")
        plt.close(fig)


def fit_continuous_interaction_regression(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """OLS regression of `metric` on the 3 continuous damage parameters, their
    pairwise products and their triple product (closed-form via
    numpy.linalg.lstsq, with standard errors/t-tests from the usual OLS
    formulas). The interaction coefficients quantify, on the continuous scale,
    the same synergistic/antagonistic effects the categorical ANOVA captures
    on the discrete LOW/MEDIUM/HIGH scale."""
    combo_df = df[df["dataset"] != "healthy"].dropna(subset=[f"{dt}_param" for dt in COMBO_DAMAGE_ORDER])
    stiffness_dt, torque_dt, gyro_dt = COMBO_DAMAGE_ORDER
    s = combo_df[f"{stiffness_dt}_param"].to_numpy()
    t = combo_df[f"{torque_dt}_param"].to_numpy()
    g = combo_df[f"{gyro_dt}_param"].to_numpy()
    y = combo_df[metric].to_numpy()

    terms = {
        "intercept": np.ones_like(s),
        stiffness_dt: s,
        torque_dt: t,
        gyro_dt: g,
        f"{stiffness_dt}:{torque_dt}": s * t,
        f"{stiffness_dt}:{gyro_dt}": s * g,
        f"{torque_dt}:{gyro_dt}": t * g,
        f"{stiffness_dt}:{torque_dt}:{gyro_dt}": s * t * g,
    }
    names = list(terms.keys())
    X = np.column_stack(list(terms.values()))

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n_obs, n_params = X.shape
    dof = n_obs - n_params
    sigma2 = np.sum(resid**2) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stat = beta / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), dof)

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n_obs - 1) / dof

    table = pd.DataFrame({"term": names, "coef": beta, "std_err": se, "t": t_stat, "p_value": p_value})
    summary = {"r_squared": float(r2), "r_squared_adj": float(r2_adj), "n_obs": int(n_obs)}
    return table, summary


def compute_and_plot_auc_per_combo(df: pd.DataFrame, out_fig: Path, metric: str) -> pd.DataFrame:
    """Bar chart + table of AUC(healthy vs combo) for each of the 27 combos,
    ordered by total severity: how detectable each combination is, rather than
    each damage type alone (a single 27-line ROC plot would be unreadable)."""
    healthy_scores = df.loc[df["dataset"] == "healthy", metric].to_numpy()
    combos = _ordered_combos(df)

    aucs: dict[str, float] = {}
    for combo in combos:
        damage_scores = df.loc[df["dataset"] == combo, metric].to_numpy()
        y_true = np.concatenate([np.zeros_like(healthy_scores), np.ones_like(damage_scores)])
        y_score = np.concatenate([healthy_scores, damage_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        aucs[combo] = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(combos, [aucs[c] for c in combos])
    ax.tick_params(axis="x", rotation=90)
    ax.axhline(0.5, linestyle=":", color="gray")
    ax.set_ylabel("AUC (healthy vs combo)")
    ax.set_title(f"Detectability per combination ({metric.upper()})")
    save_figure(fig, out_fig)
    plt.close(fig)

    return pd.DataFrame({"auc": aucs})


def export_combo_analysis(
    errors_df: pd.DataFrame,
    errors_per_feature_df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    save_table(errors_df, tables_dir / "errors_combo", formats=("csv",))
    save_table(errors_per_feature_df, tables_dir / "errors_per_feature_combo", formats=("csv",))

    for metric in METRICS:
        plot_overview_boxplot(errors_df, figures_dir / "overview_boxplot" / f"{metric}.pdf", metric)
        plot_main_effects(errors_df, figures_dir / "main_effects" / f"{metric}.pdf", metric)
        plot_interaction_2way(errors_df, figures_dir / "interaction_2way" / metric, metric)
        plot_heatmap_2way(errors_df, figures_dir / "heatmap_2way" / metric, metric)
        plot_continuous_scatter_2d(errors_df, figures_dir / "continuous_scatter_2d" / metric, metric)

        auc_table = compute_and_plot_auc_per_combo(errors_df, figures_dir / "roc_auc_per_combo" / f"{metric}.pdf", metric)
        save_table(auc_table, tables_dir / f"auc_per_combo_{metric}", formats=("csv",))

        anova_table = compute_three_way_anova(errors_df, metric)
        save_table(anova_table, tables_dir / f"anova_{metric}", formats=("csv",))

        regression_table, regression_summary = fit_continuous_interaction_regression(errors_df, metric)
        save_table(regression_table, tables_dir / f"regression_{metric}", formats=("csv",))
        save_json(regression_summary, tables_dir / f"regression_{metric}_summary.json")
