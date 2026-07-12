"""Itera run_experiment su piu' architetture e/o piu' cartelle dataset.

Un fallimento su una singola combinazione (architettura, dataset) viene loggato
e non interrompe le altre combinazioni.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import ARCHITECTURES, OPTIMIZATION_DEFAULTS, TRAINING_DEFAULTS
from .run_experiment import run_experiment
from .utils.logging_utils import get_logger


def _discover_data_folders(data_root: str) -> list[str]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cartella data_root non trovata: {root}")
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def run_all(
    architectures: list[str] | None,
    data_folders: list[str] | None,
    data_root: str = "data",
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    logger = get_logger("run_all")

    architectures = architectures or list(ARCHITECTURES)
    data_folders = data_folders or _discover_data_folders(data_root)

    logger.info("run_all: architetture=%s dataset=%s", architectures, data_folders)

    summary: list[dict[str, Any]] = []
    for data_folder in data_folders:
        for architecture in architectures:
            logger.info("--- Combinazione: architettura=%s dataset=%s ---", architecture, data_folder)
            try:
                run_experiment(architecture=architecture, data_folder=data_folder, data_root=data_root, **shared_kwargs)
                summary.append({"architecture": architecture, "data_folder": data_folder, "status": "ok"})
            except Exception as exc:
                logger.exception("Combinazione fallita: architettura=%s dataset=%s", architecture, data_folder)
                summary.append({"architecture": architecture, "data_folder": data_folder, "status": "failed", "error": str(exc)})
                continue

    n_failed = sum(1 for s in summary if s["status"] == "failed")
    logger.info("run_all completato: %d/%d combinazioni riuscite", len(summary) - n_failed, len(summary))
    for row in summary:
        logger.info("  %s", row)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Itera run_experiment su piu' architetture/dataset.")
    parser.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES), default=None, help="Default: tutte le 6")
    parser.add_argument("--data-folders", nargs="+", default=None, help="Default: tutte le sottocartelle di data-root")
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
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--no-resume-training", dest="resume_training", action="store_false")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_all(**vars(args))


if __name__ == "__main__":
    main()
