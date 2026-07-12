"""Punto unico di accesso a disco: dataset .mat v7.3, path, checkpoint, tabelle, figure."""

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
    """Costruisce e valida i path dei file .mat per una cartella dataset.

    Si aspetta ``data_root/case_name/healthy.mat`` e un file
    ``damage_<tipo>.mat`` per ciascun tipo in :data:`DAMAGE_TYPES`.
    """
    case_dir = data_root / case_name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Cartella dataset non trovata: {case_dir}")

    healthy_path = case_dir / "healthy.mat"
    if not healthy_path.is_file():
        raise FileNotFoundError(f"File healthy non trovato: {healthy_path}")

    damage_paths: dict[str, Path] = {}
    missing = []
    for damage_type in DAMAGE_TYPES:
        p = case_dir / f"damage_{damage_type}.mat"
        if not p.is_file():
            missing.append(p.name)
        damage_paths[damage_type] = p

    if missing:
        raise FileNotFoundError(
            f"File damage mancanti in {case_dir}: {', '.join(missing)}"
        )

    return DatasetPaths(healthy=healthy_path, damage=damage_paths)


def resolve_results_paths(results_root: Path, case_name: str, arch_name: str) -> ResultsPaths:
    """Costruisce (e crea su disco) l'albero di sottocartelle dei risultati."""
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
    """Le variabili MATLAB salvate in v7.3 (HDF5) sono lette da h5py con gli assi
    invertiti rispetto all'ordine logico MATLAB (colonna-maggiore vs riga-maggiore).
    Qui si inverte l'ordine degli assi per ripristinare la forma logica originale.
    """
    array = np.asarray(dataset)
    return np.transpose(array, axes=tuple(reversed(range(array.ndim))))


def load_mat_v73(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Carica le variabili richieste da un file .mat v7.3 (formato HDF5)."""
    if not path.is_file():
        raise FileNotFoundError(f"File .mat non trovato: {path}")

    result: dict[str, np.ndarray] = {}
    try:
        with h5py.File(path, "r") as f:
            for key in keys:
                if key not in f:
                    raise KeyError(f"Variabile '{key}' non presente in {path} (chiavi disponibili: {list(f.keys())})")
                result[key] = _to_matlab_order(f[key])
    except OSError as exc:
        raise OSError(f"Errore durante l'apertura/lettura di {path}: {exc}") from exc

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
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

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
        raise FileNotFoundError(f"File JSON non trovato: {path}")
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
    raise TypeError(f"Oggetto non serializzabile in JSON: {type(o)}")


def save_table(df: pd.DataFrame, path_no_suffix: Path, formats: tuple[str, ...] = ("csv",)) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        if fmt == "csv":
            df.to_csv(path_no_suffix.with_suffix(".csv"), index=True)
        elif fmt == "xlsx":
            df.to_excel(path_no_suffix.with_suffix(".xlsx"), index=True)
        else:
            raise ValueError(f"Formato tabella non supportato: {fmt}")


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
