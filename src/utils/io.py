"""Single point of access to disk: .mat v7.3 datasets, paths, checkpoints, tables, figures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAMAGE_TYPES: tuple[str, ...] = ("stiffness", "gyroscope", "torque")


@dataclass
class DatasetPaths:
    healthy: Path
    damage: dict[str, Path]


@dataclass
class ResultsPaths:
    root: Path
    preprocessing: Path
    hpo: Path
    grid_search: Path
    training: Path
    evaluation: Path
    figures: Path
    tables: Path
    logs: Path


def resolve_data_paths(data_root: Path, case_name: str) -> DatasetPaths:
    """Build and validate the .mat file paths for a dataset folder.

    Expects ``data_root/case_name/healthy.mat`` and a
    ``<type>.mat`` file for each type in :data:`DAMAGE_TYPES`.
    """
    case_dir = data_root / case_name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {case_dir}")

    healthy_path = case_dir / "healthy.mat"
    if not healthy_path.is_file():
        raise FileNotFoundError(f"Healthy file not found: {healthy_path}")

    damage_paths: dict[str, Path] = {}
    missing = []
    for damage_type in DAMAGE_TYPES:
        p = case_dir / f"{damage_type}.mat"
        if not p.is_file():
            missing.append(p.name)
        damage_paths[damage_type] = p

    if missing:
        raise FileNotFoundError(
            f"Missing damage files in {case_dir}: {', '.join(missing)}"
        )

    return DatasetPaths(healthy=healthy_path, damage=damage_paths)


def resolve_results_paths(results_root: Path, case_name: str, arch_name: str) -> ResultsPaths:
    """Build (and create on disk) the results subfolder tree."""
    root = results_root / case_name / arch_name
    paths = ResultsPaths(
        root=root,
        preprocessing=root / "preprocessing",
        hpo=root / "hpo",
        grid_search=root / "grid_search",
        training=root / "training",
        evaluation=root / "evaluation",
        figures=root / "figures",
        tables=root / "tables",
        logs=root / "logs",
    )
    for sub in (
        paths.preprocessing,
        paths.hpo,
        paths.grid_search,
        paths.training,
        paths.evaluation,
        paths.figures,
        paths.tables,
        paths.logs,
    ):
        sub.mkdir(parents=True, exist_ok=True)
    return paths


def _to_matlab_order(dataset: h5py.Dataset) -> np.ndarray:
    """MATLAB variables saved in v7.3 (HDF5) are read by h5py with axes
    reversed relative to MATLAB's logical order (column-major vs row-major).
    Here the axis order is reversed to restore the original logical shape.
    """
    array = np.asarray(dataset)
    return np.transpose(array, axes=tuple(reversed(range(array.ndim))))


def load_mat_v73(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Load the requested variables from a .mat v7.3 file (HDF5 format)."""
    if not path.is_file():
        raise FileNotFoundError(f".mat file not found: {path}")

    result: dict[str, np.ndarray] = {}
    try:
        with h5py.File(path, "r") as f:
            for key in keys:
                if key not in f:
                    raise KeyError(f"Variable '{key}' not present in {path} (available keys: {list(f.keys())})")
                result[key] = _to_matlab_order(f[key])
    except OSError as exc:
        raise OSError(f"Error while opening/reading {path}: {exc}") from exc

    return result


def save_checkpoint(path: Path, model: Any, optimizer: Any, epoch: int, best_val_loss: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, model: Any, optimizer: Any | None = None, map_location: str = "cpu") -> tuple[int, float]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object not JSON serializable: {type(o)}")


def save_table(df: pd.DataFrame, path_no_suffix: Path, formats: tuple[str, ...] = ("csv",)) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt == "csv":
            df.to_csv(path_no_suffix.with_suffix(".csv"), index=True)
        elif fmt == "xlsx":
            df.to_excel(path_no_suffix.with_suffix(".xlsx"), index=True)
        else:
            raise ValueError(f"Unsupported table format: {fmt}")


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
