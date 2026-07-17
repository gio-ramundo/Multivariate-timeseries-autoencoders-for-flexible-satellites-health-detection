"""Iterates run_experiment over multiple architectures and/or multiple dataset folders.

A failure on a single (architecture, dataset) combination is logged
and does not stop the other combinations.
"""

from __future__ import annotations

import argparse
import gc
from multiprocessing.util import info
from pathlib import Path
from typing import Any

import torch

from .config import ARCHITECTURES, OPTIMIZATION_DEFAULTS, TRAINING_DEFAULTS
from .run_experiment import run_experiment
from .utils.logging_utils import get_logger


def _discover_data_folders(data_root: str) -> list[str]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root folder not found: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def run_all(
    data_folders: list[str] | None,
    architectures: list[str] | None,
    data_root: str = "data",
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    logger = get_logger("run_all")

    architectures = architectures or list(ARCHITECTURES)
    data_folders = data_folders or _discover_data_folders(data_root)

    logger.info("run_all: architectures=%s datasets=%s", architectures, data_folders)

    summary: list[dict[str, Any]] = []
    for data_folder in data_folders:
        for architecture in architectures:
            logger.info("--- Combination: architecture=%s dataset=%s ---", architecture, data_folder)
            try:
                run_experiment(
                    architecture=architecture,
                    data_folder=data_folder,
                    data_root=data_root,
                    **shared_kwargs,
                )
                summary.append({"architecture": architecture, "data_folder": data_folder, "status": "ok"})
            except Exception as exc:
                logger.exception("Combination failed: architecture=%s dataset=%s", architecture, data_folder)
                summary.append({"architecture": architecture, "data_folder": data_folder, "status": "failed", "error": str(exc)})
                continue
            finally:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("VRAM cache freed after architecture=%s dataset=%s", architecture, data_folder)

    n_failed = sum(1 for s in summary if s["status"] == "failed")
    logger.info("run_all completed: %d/%d combinations succeeded", len(summary) - n_failed, len(summary))
    for row in summary:
        logger.info("  %s", row)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterates run_experiment over multiple architectures/datasets.")
    parser.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES), default=None, help="Default: all 6")
    parser.add_argument("--data-folders", nargs="+", default=None, help="Default: all subfolders of data-root")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--n-hpo-trials", type=int, default=OPTIMIZATION_DEFAULTS["n_hpo_trials"])
    parser.add_argument("--top-n", type=int, default=OPTIMIZATION_DEFAULTS["top_n"])
    parser.add_argument("--parsimony-tolerance", type=float, default=OPTIMIZATION_DEFAULTS["parsimony_tolerance"])
    parser.add_argument("--hpo-epochs", type=int, default=TRAINING_DEFAULTS["hpo_epochs"])
    parser.add_argument("--grid-epochs", type=int, default=TRAINING_DEFAULTS["grid_epochs"])
    parser.add_argument("--grid-resolution", type=int, default=OPTIMIZATION_DEFAULTS["grid_resolution"])
    parser.add_argument("--grid-max-combinations", type=int, default=OPTIMIZATION_DEFAULTS["grid_max_combinations"])
    parser.add_argument("--final-epochs", type=int, default=TRAINING_DEFAULTS["final_epochs"])
    parser.add_argument("--seed", type=int, default=OPTIMIZATION_DEFAULTS["seed"])
    parser.add_argument(
        "--n-jobs-hpo",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Parallel HPO trials (threads), applied identically to every (architecture, dataset) "
            "combination. One value per sub-run: the full-length run first, then each chunk length "
            "in the order given to --chunk-lengths. Give 1 value to apply it to every sub-run, or "
            "exactly (1 + number of --chunk-lengths) values."
        ),
    )
    parser.add_argument(
        "--n-jobs-gs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Parallel grid search combinations (threads), applied identically to every (architecture, "
            "dataset) combination. One value per sub-run: the full-length run first, then each chunk "
            "length in the order given to --chunk-lengths. Give 1 value to apply it to every sub-run, "
            "or exactly (1 + number of --chunk-lengths) values."
        ),
    )
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--no-resume-training", dest="resume_training", action="store_false")
    parser.add_argument(
        "--chunk-lengths",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Chunk lengths (timesteps) to additionally train/evaluate on, besides the full-length "
            "run, for every combination. Windows overlap by 20%% of the chunk length. "
            "Default: none (full length only)."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_all(**vars(args))


if __name__ == "__main__":
    main()
