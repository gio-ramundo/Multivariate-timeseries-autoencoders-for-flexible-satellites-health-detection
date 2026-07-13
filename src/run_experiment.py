"""Runs the full pipeline (preprocessing -> Bayesian HPO -> grid search ->
final training/test -> results export) for ONE (architecture, dataset) pair.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .bayesian_optimizer import run_bayesian_optimization
from .config import ARCHITECTURES, OPTIMIZATION_DEFAULTS, TRAINING_DEFAULTS
from .grid_search import run_grid_search
from .preprocessing import run_preprocessing
from .results_export import export_all_results
from .train_test import run_train_test
from .utils.io import resolve_data_paths, resolve_results_paths
from .utils.logging_utils import get_logger


def run_experiment(
    architecture: str,
    data_folder: str,
    data_root: str = "data",
    results_root: str = "results",
    n_hpo_trials: int = OPTIMIZATION_DEFAULTS["n_hpo_trials"],
    top_n: int = OPTIMIZATION_DEFAULTS["top_n"],
    parsimony_tolerance: float = OPTIMIZATION_DEFAULTS["parsimony_tolerance"],
    hpo_epochs: int = TRAINING_DEFAULTS["hpo_epochs"],
    grid_epochs: int = TRAINING_DEFAULTS["grid_epochs"],
    grid_resolution: int = OPTIMIZATION_DEFAULTS["grid_resolution"],
    grid_max_combinations: int = OPTIMIZATION_DEFAULTS["grid_max_combinations"],
    final_epochs: int = TRAINING_DEFAULTS["final_epochs"],
    seed: int = OPTIMIZATION_DEFAULTS["seed"],
    n_jobs_hpo: int = OPTIMIZATION_DEFAULTS["n_jobs_hpo"],
    n_jobs_gs: int = OPTIMIZATION_DEFAULTS["n_jobs_gs"],
    force_preprocessing: bool = False,
    resume_training: bool = True,
) -> dict[str, Any]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(ARCHITECTURES)}")

    logger = get_logger("run_experiment")
    dataset_paths = resolve_data_paths(Path(data_root), data_folder)
    results_paths = resolve_results_paths(Path(results_root), data_folder, architecture)

    logger = get_logger(f"run_experiment.{data_folder}.{architecture}", results_paths.logs / "run_experiment.log")

    try:
        logger.info("=== Starting experiment: architecture=%s dataset=%s ===", architecture, data_folder)

        preprocessed = run_preprocessing(dataset_paths, results_paths, seed=seed, force=force_preprocessing)

        narrowed_ranges = run_bayesian_optimization(
            architecture,
            preprocessed,
            results_paths,
            n_trials=n_hpo_trials,
            top_n=top_n,
            hpo_epochs=hpo_epochs,
            tolerance=parsimony_tolerance,
            seed=seed,
            n_jobs=n_jobs_hpo,
        )

        best_hp = run_grid_search(
            architecture,
            preprocessed,
            narrowed_ranges,
            results_paths,
            epochs=grid_epochs,
            resolution=grid_resolution,
            max_combinations=grid_max_combinations,
            seed=seed,
            n_jobs=n_jobs_gs,
        )

        train_test_results = run_train_test(
            architecture, best_hp, preprocessed, results_paths, epochs=final_epochs, resume=resume_training, seed=seed
        )

        export_all_results(preprocessed, train_test_results, results_paths)

        logger.info("=== Experiment completed: architecture=%s dataset=%s ===", architecture, data_folder)
        return {"results_paths": results_paths, "best_hyperparams": best_hp, "train_test_results": train_test_results}
    except Exception:
        logger.exception("Experiment failed: architecture=%s dataset=%s", architecture, data_folder)
        raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runs the full pipeline for one (architecture, dataset) pair.")
    parser.add_argument("--architecture", required=True, choices=list(ARCHITECTURES))
    parser.add_argument("--data-folder", required=True)
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
    parser.add_argument("--n-jobs-hpo", type=int, default=OPTIMIZATION_DEFAULTS["n_jobs_hpo"], help="Parallel HPO trials (threads)")
    parser.add_argument("--n-jobs-gs", type=int, default=OPTIMIZATION_DEFAULTS["n_jobs_gs"], help="Parallel grid search combinations (threads)")
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--no-resume-training", dest="resume_training", action="store_false")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
