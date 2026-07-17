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


def _resolve_per_dataset(values: list[int], n_datasets: int, name: str) -> list[int]:
    """Broadcasts a single value to all datasets, or matches a per-dataset list 1:1
    (in the same order as `data_folders`)."""
    if len(values) == 1:
        return values * n_datasets
    if len(values) != n_datasets:
        raise ValueError(
            f"--{name} must be given either 1 value (applied to all datasets) or exactly "
            f"{n_datasets} values (one per dataset, in the same order), got {len(values)}: {values}"
        )
    return values


def run_all(
    data_folders: list[str] | None,
    architectures: list[str] | None,
    data_root: str = "data",
    n_jobs_hpo: list[int] | None = None,
    n_jobs_gs: list[int] | None = None,
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    logger = get_logger("run_all")

    architectures = architectures or list(ARCHITECTURES)
    data_folders = data_folders or _discover_data_folders(data_root)

    n_jobs_hpo = n_jobs_hpo if n_jobs_hpo is not None else [OPTIMIZATION_DEFAULTS["n_jobs_hpo"]]
    n_jobs_gs = n_jobs_gs if n_jobs_gs is not None else [OPTIMIZATION_DEFAULTS["n_jobs_gs"]]
    n_jobs_hpo_by_dataset = _resolve_per_dataset(n_jobs_hpo, len(data_folders), "n-jobs-hpo")
    n_jobs_gs_by_dataset = _resolve_per_dataset(n_jobs_gs, len(data_folders), "n-jobs-gs")

    logger.info("run_all: architectures=%s datasets=%s", architectures, data_folders)
    for data_folder, jh, jg in zip(data_folders, n_jobs_hpo_by_dataset, n_jobs_gs_by_dataset):
        logger.info("  dataset=%s -> n_jobs_hpo=%d n_jobs_gs=%d", data_folder, jh, jg)

    summary: list[dict[str, Any]] = []
    for data_folder, jobs_hpo, jobs_gs in zip(data_folders, n_jobs_hpo_by_dataset, n_jobs_gs_by_dataset):
        for architecture in architectures:
            logger.info("--- Combination: architecture=%s dataset=%s ---", architecture, data_folder)
            try:
                run_experiment(
                    architecture=architecture,
                    data_folder=data_folder,
                    data_root=data_root,
                    n_jobs_hpo=jobs_hpo,
                    n_jobs_gs=jobs_gs,
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
        default=1,
        help=(
            "Parallel HPO trials (threads). Give 1 value to apply it to all datasets, or one value "
            "per dataset (same order as --data-folders), e.g. --data-folders A B --n-jobs-hpo 2 5"
        ),
    )
    parser.add_argument(
        "--n-jobs-gs",
        type=int,
        nargs="+",
        default=1,
        help=(
            "Parallel grid search combinations (threads). Give 1 value to apply it to all datasets, or "
            "one value per dataset (same order as --data-folders), e.g. --data-folders A B --n-jobs-gs 2 5"
        ),
    )
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--no-resume-training", dest="resume_training", action="store_false")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_all(**vars(args))


if __name__ == "__main__":
    main()
