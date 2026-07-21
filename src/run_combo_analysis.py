"""Inference-only interaction analysis: given a combo data folder (one healthy
file + any number of LOW/MEDIUM/HIGH stiffness/torque/gyroscope combination
files, see `combo_data.py`), reuses the already-trained model, normalization
stats and best hyperparameters of a prior single-damage experiment (run via
`run_experiment.py`) to run preprocessing + inference + error computation, and
exports the interaction analysis in `combo_stats.py`.

No training happens here: this script only loads what `run_experiment.py`
already produced. It is run once per (architecture, run variant) pair, where a
"run variant" is either the full-length model or one of its chunked
counterparts (e.g. "<base_experiment>-chunk_500"), mirroring how
`run_experiment.py` organizes those as separate results folders.
"""

from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .combo_data import ComboDataset, LoadedCombo, discover_combo_dataset, load_combo_dataset
from .combo_stats import build_combo_errors_table, export_combo_analysis
from .config import ArchitectureSpec, ARCHITECTURES
from .models import Autoencoder
from .preprocessing import NormalizationStats
from .train_test import compute_errors, run_chunked_inference, run_inference
from .utils.io import load_checkpoint, load_json
from .utils.logging_utils import get_logger

logger = get_logger("run_combo_analysis")

# Every architecture variant ever defined for this project (see config.py),
# including ones currently commented out of config.ARCHITECTURES. Kept here,
# duplicated from those comments, only so this analysis script can run
# inference against models trained back when a now-disabled architecture was
# still active, without touching config.py (whose ARCHITECTURES dict reflects
# what should be *trained going forward*, not what has ever been trained).
_ALL_ARCHITECTURE_SPECS: dict[str, ArchitectureSpec] = {
    "conv_lstm_vec": ArchitectureSpec("conv_lstm_vec", n_conv_layers=1, use_activation=False, latent_mode="vector"),
    "conv_lstm_seq": ArchitectureSpec("conv_lstm_seq", n_conv_layers=1, use_activation=False, latent_mode="sequence"),
    "conv_relu_lstm_vec": ArchitectureSpec("conv_relu_lstm_vec", n_conv_layers=1, use_activation=True, latent_mode="vector"),
    "conv_relu_lstm_seq": ArchitectureSpec("conv_relu_lstm_seq", n_conv_layers=1, use_activation=True, latent_mode="sequence"),
    "conv2_relu_lstm_vec": ArchitectureSpec("conv2_relu_lstm_vec", n_conv_layers=2, use_activation=True, latent_mode="vector"),
    "conv2_relu_lstm_seq": ArchitectureSpec("conv2_relu_lstm_seq", n_conv_layers=2, use_activation=True, latent_mode="sequence"),
}


def _architecture_spec(arch_name: str) -> ArchitectureSpec:
    if arch_name in ARCHITECTURES:
        return ARCHITECTURES[arch_name]
    if arch_name in _ALL_ARCHITECTURE_SPECS:
        return _ALL_ARCHITECTURE_SPECS[arch_name]
    raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(_ALL_ARCHITECTURE_SPECS)}")


@dataclass
class ComboResultsPaths:
    root: Path
    figures: Path
    tables: Path


def _resolve_combo_results_paths(results_root: Path, case_name: str, arch_name: str) -> ComboResultsPaths:
    """Results layout for this analysis: evaluation/figures/tables/logs only
    (no hpo/grid_search/training subfolders, since nothing is trained here),
    mirroring the same <case_name>/<arch_name> convention `run_experiment.py`
    uses so the two analyses sit side by side under `results/`."""
    root = results_root / case_name / arch_name
    paths = ComboResultsPaths(root=root, figures=root / "figures", tables=root / "tables")
    for sub in (paths.figures, paths.tables):
        sub.mkdir(parents=True, exist_ok=True)
    return paths


def _discover_architectures(results_root: Path, base_experiment: str) -> list[str]:
    base_dir = results_root / base_experiment
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base experiment results folder not found: {base_dir}")
    return sorted(
        p.name for p in base_dir.iterdir()
        if p.is_dir() and (p / "training" / "best_model.pt").is_file()
    )


def _discover_chunk_lengths(results_root: Path, base_experiment: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(base_experiment)}-chunk_(\d+)$")
    lengths = []
    for p in results_root.iterdir():
        if p.is_dir() and (m := pattern.match(p.name)):
            lengths.append(int(m.group(1)))
    return sorted(lengths)


def _load_model(
    arch_name: str, model_case_dir: Path, input_len: int, n_features: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    hp = load_json(model_case_dir / "grid_search" / "best_hyperparams.json")
    model = Autoencoder(_architecture_spec(arch_name), hp, input_len, n_features).to(device)
    load_checkpoint(model_case_dir / "training" / "best_model.pt", model, map_location=str(device))
    model.eval()
    return model, hp


def _run_one(
    dataset: ComboDataset,
    base_experiment: str,
    model_case_name: str,
    data_case_name: str,
    arch_name: str,
    chunk_len: int | None,
    results_root: Path,
) -> None:
    label = f"{data_case_name}/{arch_name}"
    logger.info("=== Combo analysis: %s (model from %s) ===", label, model_case_name)

    norm_stats_path = results_root / base_experiment / arch_name / "preprocessing" / "norm_stats.json"
    norm_stats = NormalizationStats.from_dict(load_json(norm_stats_path))
    loaded: LoadedCombo = load_combo_dataset(dataset, norm_stats)

    n_features = loaded.healthy.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_case_dir = results_root / model_case_name / arch_name
    input_len = chunk_len if chunk_len is not None else loaded.healthy.shape[1]
    model, hp = _load_model(arch_name, model_case_dir, input_len, n_features, device)
    batch_size = hp["batch_size"]

    def infer(x):
        if chunk_len is None:
            return run_inference(model, x, device, batch_size=batch_size)
        return run_chunked_inference(model, x, chunk_len, device, batch_size=batch_size)

    recon_healthy, _ = infer(loaded.healthy)
    healthy_total, healthy_per_feature = compute_errors(loaded.healthy, recon_healthy)

    combo_totals: dict[str, Any] = {}
    combo_per_feature: dict[str, Any] = {}
    for combo, data in loaded.combo_data.items():
        recon, _ = infer(data)
        total_df, per_feature_df = compute_errors(data, recon)
        combo_totals[combo] = total_df
        combo_per_feature[combo] = per_feature_df

    errors_df = build_combo_errors_table(healthy_total, combo_totals, loaded.combo_levels, loaded.combo_params)
    errors_per_feature_df = build_combo_errors_table(healthy_per_feature, combo_per_feature, loaded.combo_levels, loaded.combo_params)

    results_paths = _resolve_combo_results_paths(results_root, data_case_name, arch_name)
    export_combo_analysis(errors_df, errors_per_feature_df, results_paths.figures, results_paths.tables)

    logger.info("=== Combo analysis completed: %s ===", label)


def run_combo_analysis(
    data_folder: str,
    base_experiment: str | None = None,
    data_root: str = "data",
    results_root: str = "results",
    architectures: list[str] | None = None,
    chunk_lengths: list[int] | None = None,
    include_chunks: bool = True,
) -> None:
    data_root_p = Path(data_root)
    results_root_p = Path(results_root)
    base_experiment = base_experiment or data_folder.split("_")[0]

    dataset = discover_combo_dataset(data_root_p / data_folder)

    if architectures is None:
        architectures = _discover_architectures(results_root_p, base_experiment)
        logger.info("Auto-discovered architectures with a trained model in '%s': %s", base_experiment, architectures)
    if not architectures:
        raise ValueError(f"No trained architecture found under results/{base_experiment}")

    if chunk_lengths is None:
        chunk_lengths = _discover_chunk_lengths(results_root_p, base_experiment) if include_chunks else []
        logger.info("Auto-discovered chunk lengths for '%s': %s", base_experiment, chunk_lengths)

    variants: list[tuple[str, int | None]] = [(base_experiment, None)]
    variants += [(f"{base_experiment}-chunk_{cl}", cl) for cl in chunk_lengths]

    for model_case_name, chunk_len in variants:
        data_case_name = data_folder if chunk_len is None else f"{data_folder}-chunk_{chunk_len}"
        for arch_name in architectures:
            try:
                _run_one(dataset, base_experiment, model_case_name, data_case_name, arch_name, chunk_len, results_root_p)
            except Exception:
                logger.exception("Combo analysis failed for %s/%s (model %s)", data_case_name, arch_name, model_case_name)
                continue
            finally:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference-only interaction analysis of a multi-damage combo dataset against a pretrained model."
    )
    parser.add_argument("--data-folder", required=True, help="Folder under data-root holding the combo .mat files.")
    parser.add_argument(
        "--base-experiment",
        default=None,
        help="Folder under results-root holding the pretrained model/norm-stats/hyperparameters (as produced by "
        "run_experiment.py). Default: the part of --data-folder before the first underscore.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--architectures", nargs="+", default=None,
        help="Default: every architecture with a saved model under results/<base-experiment>.",
    )
    parser.add_argument(
        "--chunk-lengths", type=int, nargs="+", default=None,
        help="Default: every chunk length with a results/<base-experiment>-chunk_<length> folder.",
    )
    parser.add_argument("--no-chunks", dest="include_chunks", action="store_false", help="Skip chunked variants entirely.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_combo_analysis(**vars(args))


if __name__ == "__main__":
    main()
