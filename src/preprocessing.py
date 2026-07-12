"""Selezione feature, split healty train/val/test, normalizzazione z-score.

La normalizzazione e' calcolata esclusivamente sul training set healthy e
applicata (senza ricalcolo) a validation, test e a tutti i dataset damage,
per evitare data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils.io import DatasetPaths, ResultsPaths, load_mat_v73
from .utils.logging_utils import get_logger

FEATURE_INDICES: list[int] = [0, 1, 2, 6, 7, 8, 9, 10, 11, 18, 19, 20]
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

logger = get_logger(__name__)


@dataclass
class NormalizationStats:
    mean: np.ndarray  # shape (n_features,)
    std: np.ndarray  # shape (n_features,)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "NormalizationStats":
        return NormalizationStats(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]))


@dataclass
class PreprocessedData:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    damage: dict[str, np.ndarray]
    damage_parameter: dict[str, np.ndarray]
    norm_stats: NormalizationStats


def select_features(data: np.ndarray) -> np.ndarray:
    """Seleziona le colonne feature in FEATURE_INDICES sull'ultimo asse."""
    n_features = data.shape[-1]
    max_idx = max(FEATURE_INDICES)
    if max_idx >= n_features:
        raise ValueError(
            f"FEATURE_INDICES richiede almeno {max_idx + 1} feature, "
            f"ma il dataset ne ha {n_features}"
        )
    return data[..., FEATURE_INDICES]


def split_healthy(
    n_instances: int, seed: int, ratios: tuple[float, float, float] = SPLIT_RATIOS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split per istanza (non temporale) in train/val/test, con permutazione seedata."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Le ratios di split devono sommare a 1.0, ricevuto {ratios}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_instances)

    n_train = int(round(ratios[0] * n_instances))
    n_val = int(round(ratios[1] * n_instances))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def fit_normalization(train_data: np.ndarray) -> NormalizationStats:
    """Media/std per feature calcolate su tutte le istanze e gli istanti temporali del train."""
    mean = train_data.mean(axis=(0, 1))
    std = train_data.std(axis=(0, 1))
    std_safe = np.where(std < 1e-8, 1.0, std)
    if np.any(std < 1e-8):
        logger.warning("Deviazione standard ~0 per alcune feature: normalizzazione impostata a 1.0 per quelle feature")
    return NormalizationStats(mean=mean, std=std_safe)


def apply_normalization(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (data - stats.mean) / stats.std


def _squeeze_damage_parameter(raw: np.ndarray, n_instances: int) -> np.ndarray:
    arr = np.asarray(raw).squeeze()
    if arr.ndim != 1 or arr.shape[0] != n_instances:
        raise ValueError(
            f"damage_parameter ha shape inattesa {np.asarray(raw).shape} "
            f"per {n_instances} istanze"
        )
    return arr


def _cache_paths(results_paths: ResultsPaths) -> tuple[Path, Path]:
    return (
        results_paths.preprocessing / "preprocessed.npz",
        results_paths.preprocessing / "norm_stats.json",
    )


def run_preprocessing(
    dataset_paths: DatasetPaths, results_paths: ResultsPaths, seed: int = 0, force: bool = False
) -> PreprocessedData:
    """Esegue (o carica dalla cache) l'intero preprocessing per una cartella dataset."""
    from .utils.io import load_json, save_json

    npz_path, stats_path = _cache_paths(results_paths)

    if not force and npz_path.is_file() and stats_path.is_file():
        try:
            logger.info("Preprocessing in cache trovato, carico da %s", npz_path)
            cached = np.load(npz_path, allow_pickle=False)
            stats = NormalizationStats.from_dict(load_json(stats_path))
            damage = {dt: cached[f"damage_{dt}"] for dt in ("stiffness", "gyroscope", "torque")}
            damage_parameter = {
                dt: cached[f"damage_parameter_{dt}"] for dt in ("stiffness", "gyroscope", "torque")
            }
            return PreprocessedData(
                train=cached["train"],
                val=cached["val"],
                test=cached["test"],
                damage=damage,
                damage_parameter=damage_parameter,
                norm_stats=stats,
            )
        except Exception:
            logger.exception("Cache di preprocessing corrotta o incompleta, rieseguo da zero")

    try:
        logger.info("Carico dataset healthy da %s", dataset_paths.healthy)
        healthy_raw = load_mat_v73(dataset_paths.healthy, ["data"])["data"]
        if healthy_raw.ndim != 3:
            raise ValueError(f"'data' healthy deve avere 3 dimensioni, trovate {healthy_raw.ndim}")

        healthy = select_features(healthy_raw)
        n_instances = healthy.shape[0]
        logger.info("Dataset healthy: %d istanze, %d timestep, %d feature selezionate", *healthy.shape)

        train_idx, val_idx, test_idx = split_healthy(n_instances, seed)
        logger.info("Split healthy -> train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

        stats = fit_normalization(healthy[train_idx])

        train = apply_normalization(healthy[train_idx], stats)
        val = apply_normalization(healthy[val_idx], stats)
        test = apply_normalization(healthy[test_idx], stats)

        damage: dict[str, np.ndarray] = {}
        damage_parameter: dict[str, np.ndarray] = {}
        for damage_type, path in dataset_paths.damage.items():
            logger.info("Carico dataset damage '%s' da %s", damage_type, path)
            raw = load_mat_v73(path, ["data", "damage_parameter"])
            d_data = select_features(raw["data"])
            d_param = _squeeze_damage_parameter(raw["damage_parameter"], d_data.shape[0])
            damage[damage_type] = apply_normalization(d_data, stats)
            damage_parameter[damage_type] = d_param
            logger.info("Dataset damage '%s': %d istanze", damage_type, d_data.shape[0])

        save_json(stats.to_dict(), stats_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            train=train,
            val=val,
            test=test,
            **{f"damage_{k}": v for k, v in damage.items()},
            **{f"damage_parameter_{k}": v for k, v in damage_parameter.items()},
        )
        logger.info("Preprocessing completato e salvato in %s", npz_path)

        return PreprocessedData(
            train=train, val=val, test=test, damage=damage, damage_parameter=damage_parameter, norm_stats=stats
        )
    except Exception:
        logger.exception("Preprocessing fallito per il dataset in %s", dataset_paths.healthy.parent)
        raise
