"""Runs the full pipeline (preprocessing -> Bayesian HPO -> grid search ->
final training/test -> results export) for ONE (architecture, dataset) pair.

If `chunk_lengths` is given, the same pipeline is additionally run once per
chunk length: the model is trained on overlapping windows of the series
instead of the full length, and at inference time the per-window
reconstructions are stitched back into full-length series before computing
the same statistics as the full-length run. Each chunk length gets its own
results folder (data_folder-chunk_<length>) and failures on one chunk length
do not stop the others or the full-length run.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from .bayesian_optimizer import run_bayesian_optimization
from .config import ARCHITECTURES, OPTIMIZATION_DEFAULTS, TRAINING_DEFAULTS, CHUNK_OVERLAP_RATIO
from .grid_search import run_grid_search
from .preprocessing import PreprocessedData, chunk_preprocessed, run_preprocessing
from .results_export import export_all_results
from .train_test import run_train_test
from .utils.io import ResultsPaths, resolve_data_paths, resolve_results_paths
from .utils.logging_utils import get_logger


def _resolve_per_run(values: list[int], n_runs: int, name: str) -> list[int]:
    """Broadcasts a single value to every sub-run (the full-length run, then each
    chunk length in the order given to --chunk-lengths), or matches a per-run list
    1:1."""
    if len(values) == 1:
        return values * n_runs
    if len(values) != n_runs:
        raise ValueError(
            f"--{name} must be given either 1 value (applied to the full-length run and every chunk "
            f"length) or exactly {n_runs} values (one per sub-run: full-length first, then each chunk "
            f"length in the order given to --chunk-lengths), got {len(values)}: {values}"
        )
    return values


def _run_pipeline(
    architecture: str,
    data: PreprocessedData,
    results_paths: ResultsPaths,
    n_hpo_trials: int,
    top_n: int,
    parsimony_tolerance: float,
    hpo_epochs: int,
    grid_epochs: int,
    grid_resolution: int,
    grid_max_combinations: int,
    final_epochs: int,
    seed: int,
    n_jobs_hpo: int,
    n_jobs_gs: int,
    resume_training: bool,
    chunk_len: int | None = None,
    chunk_overlap: float = CHUNK_OVERLAP_RATIO,
) -> dict[str, Any]:
    """HPO -> grid search -> final train/test -> export, for one already-preprocessed
    (and, for a chunk run, already-chunked) dataset. `chunk_len`/`chunk_overlap` only
    affect evaluation: they tell the final train/test step to stitch per-window
    reconstructions back into full-length series before computing statistics."""
    narrowed_ranges = run_bayesian_optimization(
        architecture,
        data,
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
        data,
        narrowed_ranges,
        results_paths,
        epochs=grid_epochs,
        resolution=grid_resolution,
        max_combinations=grid_max_combinations,
        seed=seed,
        n_jobs=n_jobs_gs,
    )

    train_test_results = run_train_test(
        architecture,
        best_hp,
        data,
        results_paths,
        epochs=final_epochs,
        resume=resume_training,
        seed=seed,
        chunk_len=chunk_len,
        chunk_overlap=chunk_overlap,
    )

    export_all_results(data, train_test_results, results_paths, seed=seed)

    return {"results_paths": results_paths, "best_hyperparams": best_hp, "train_test_results": train_test_results}


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
    n_jobs_hpo: list[int] | None = None,
    n_jobs_gs: list[int] | None = None,
    force_preprocessing: bool = False,
    resume_training: bool = True,
    chunk_lengths: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(ARCHITECTURES)}")

    chunk_lengths = chunk_lengths or []
    n_runs = 1 + len(chunk_lengths)

    n_jobs_hpo = n_jobs_hpo if n_jobs_hpo is not None else [OPTIMIZATION_DEFAULTS["n_jobs_hpo"]]
    n_jobs_gs = n_jobs_gs if n_jobs_gs is not None else [OPTIMIZATION_DEFAULTS["n_jobs_gs"]]
    n_jobs_hpo_by_run = _resolve_per_run(n_jobs_hpo, n_runs, "n-jobs-hpo")
    n_jobs_gs_by_run = _resolve_per_run(n_jobs_gs, n_runs, "n-jobs-gs")

    logger = get_logger("run_experiment")
    dataset_paths = resolve_data_paths(Path(data_root), data_folder)
    results_paths = resolve_results_paths(Path(results_root), data_folder, architecture)

    logger = get_logger(f"run_experiment.{data_folder}.{architecture}", results_paths.logs / "run_experiment.log")

    pipeline_kwargs = dict(
        n_hpo_trials=n_hpo_trials,
        top_n=top_n,
        parsimony_tolerance=parsimony_tolerance,
        hpo_epochs=hpo_epochs,
        grid_epochs=grid_epochs,
        grid_resolution=grid_resolution,
        grid_max_combinations=grid_max_combinations,
        final_epochs=final_epochs,
        seed=seed,
        resume_training=resume_training,
    )

    try:
        logger.info("=== Starting experiment: architecture=%s dataset=%s ===", architecture, data_folder)

        preprocessed = run_preprocessing(dataset_paths, results_paths, seed=seed, force=force_preprocessing)

        runs: dict[str, dict[str, Any]] = {
            "full": _run_pipeline(
                architecture, preprocessed, results_paths,
                n_jobs_hpo=n_jobs_hpo_by_run[0], n_jobs_gs=n_jobs_gs_by_run[0], **pipeline_kwargs,
            )
        }

        logger.info("=== Experiment completed (full-length): architecture=%s dataset=%s ===", architecture, data_folder)

        for i, chunk_len in enumerate(chunk_lengths, start=1):
            chunk_case_name = f"{data_folder}-chunk_{chunk_len}"
            chunk_results_paths = resolve_results_paths(Path(results_root), chunk_case_name, architecture)
            chunk_logger = get_logger(
                f"run_experiment.{chunk_case_name}.{architecture}", chunk_results_paths.logs / "run_experiment.log"
            )
            chunk_logger.info(
                "=== Starting chunked run: architecture=%s dataset=%s chunk_len=%d ===", architecture, data_folder, chunk_len
            )
            try:
                chunk_data = chunk_preprocessed(preprocessed, chunk_len)
                runs[f"chunk_{chunk_len}"] = _run_pipeline(
                    architecture, chunk_data, chunk_results_paths,
                    n_jobs_hpo=n_jobs_hpo_by_run[i], n_jobs_gs=n_jobs_gs_by_run[i],
                    chunk_len=chunk_len, **pipeline_kwargs,
                )
                chunk_logger.info(
                    "=== Chunked run completed: architecture=%s dataset=%s chunk_len=%d ===", architecture, data_folder, chunk_len
                )
            except Exception:
                chunk_logger.exception(
                    "Chunked run failed: architecture=%s dataset=%s chunk_len=%d", architecture, data_folder, chunk_len
                )
            finally:
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

        logger.info("=== Experiment completed: architecture=%s dataset=%s ===", architecture, data_folder)
        return runs
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
    parser.add_argument(
        "--n-jobs-hpo",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Parallel HPO trials (threads), one value per sub-run: the full-length run first, then "
            "each chunk length in the order given to --chunk-lengths. Give 1 value to apply it to "
            "every sub-run, or exactly (1 + number of --chunk-lengths) values."
        ),
    )
    parser.add_argument(
        "--n-jobs-gs",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Parallel grid search combinations (threads), one value per sub-run: the full-length run "
            "first, then each chunk length in the order given to --chunk-lengths. Give 1 value to "
            "apply it to every sub-run, or exactly (1 + number of --chunk-lengths) values."
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
            "run. Windows overlap by 20%% of the chunk length. Default: none (full length only)."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_experiment(**vars(args))


if __name__ == "__main__":
    main()
