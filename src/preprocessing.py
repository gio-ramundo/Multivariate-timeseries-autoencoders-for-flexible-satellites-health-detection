"""Feature selection, healthy train/val/test split, z-score normalization.

Normalization is computed exclusively on the healthy training set and
applied (without recomputation) to validation, test and all damage
datasets, in order to avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils.io import DatasetPaths, ResultsPaths, load_mat_v73
from .utils.logging_utils import get_logger

FEATURE_INDICES: list[int] = [3, 4, 5, 9, 10, 11, 17, 18, 19, 20, 21, 22, 23, 24, 25]
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

# Object names inside the .mat files. The healthy file holds a single 3D
# array; each damage file holds a 3D array plus a 1D damage-level array
# named after the damage type itself (e.g. "stiffness", not "damage_parameter").
HEALTHY_DATA_KEY = "dataset_healthy"


def _damage_data_key(damage_type: str) -> str:
    if damage_type == "gyroscope":
        return "dataset_gyro"
    return f"dataset_{damage_type}"


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


def reorder_to_instance_first(data: np.ndarray) -> np.ndarray:
    """Reorder a raw .mat array from MATLAB's logical layout
    (timestep, features, samples) to the (samples, timestep, features)
    convention used throughout the rest of the pipeline (models, splitting,
    error computation all index the first axis as the instance/batch axis).
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D array (timestep, features, samples), got ndim={data.ndim}")
    return np.transpose(data, axes=(2, 0, 1))


def select_features(data: np.ndarray) -> np.ndarray:
    """Select the feature columns in FEATURE_INDICES along the last axis."""
    n_features = data.shape[-1]
    max_idx = max(FEATURE_INDICES)
    if max_idx >= n_features:
        raise ValueError(
            f"FEATURE_INDICES requires at least {max_idx + 1} features, "
            f"but the dataset has {n_features}"
        )
    return data[..., FEATURE_INDICES]


def split_healthy(
    n_instances: int, seed: int, ratios: tuple[float, float, float] = SPLIT_RATIOS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-instance (non-temporal) split into train/val/test, with a seeded permutation."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_instances)

    n_train = int(round(ratios[0] * n_instances))
    n_val = int(round(ratios[1] * n_instances))

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def fit_normalization(train_data: np.ndarray) -> NormalizationStats:
    """Per-feature mean/std computed over all training instances and timesteps."""
    mean = train_data.mean(axis=(0, 1))
    std = train_data.std(axis=(0, 1))
    std_safe = np.where(std < 1e-8, 1.0, std)
    if np.any(std < 1e-8):
        logger.warning("Standard deviation ~0 for some features: normalization set to 1.0 for those features")
    return NormalizationStats(mean=mean, std=std_safe)


def apply_normalization(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (data - stats.mean) / stats.std


def _squeeze_damage_parameter(raw: np.ndarray, n_instances: int) -> np.ndarray:
    arr = np.asarray(raw).squeeze()
    if arr.ndim != 1 or arr.shape[0] != n_instances:
        raise ValueError(
            f"damage_parameter has unexpected shape {np.asarray(raw).shape} "
            f"for {n_instances} instances"
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
    """Run (or load from cache) the full preprocessing for a dataset folder."""
    from .utils.io import load_json, save_json

    npz_path, stats_path = _cache_paths(results_paths)

    if not force and npz_path.is_file() and stats_path.is_file():
        try:
            logger.info("Cached preprocessing found, loading from %s", npz_path)
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
            logger.exception("Preprocessing cache corrupted or incomplete, re-running from scratch")

    try:
        logger.info("Loading healthy dataset from %s", dataset_paths.healthy)
        healthy_raw = load_mat_v73(dataset_paths.healthy, [HEALTHY_DATA_KEY])[HEALTHY_DATA_KEY]
        if healthy_raw.ndim != 3:
            raise ValueError(f"Healthy '{HEALTHY_DATA_KEY}' must have 3 dimensions, found {healthy_raw.ndim}")

        healthy = select_features(reorder_to_instance_first(healthy_raw))
        ## for test
        #healthy = healthy[:, :200, :]
        n_instances = healthy.shape[0]
        logger.info("Healthy dataset: %d instances, %d timesteps, %d selected features", *healthy.shape)

        train_idx, val_idx, test_idx = split_healthy(n_instances, seed)
        logger.info("Healthy split -> train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

        stats = fit_normalization(healthy[train_idx])

        train = apply_normalization(healthy[train_idx], stats)
        val = apply_normalization(healthy[val_idx], stats)
        test = apply_normalization(healthy[test_idx], stats)

        damage: dict[str, np.ndarray] = {}
        damage_parameter: dict[str, np.ndarray] = {}
        for damage_type, path in dataset_paths.damage.items():
            logger.info("Loading damage dataset '%s' from %s", damage_type, path)
            data_key = _damage_data_key(damage_type)
            raw = load_mat_v73(path, [data_key, damage_type])
            d_data = select_features(reorder_to_instance_first(raw[data_key]))
            ## for test
            #d_data = d_data[:, :200, :]
            d_param = _squeeze_damage_parameter(raw[damage_type], d_data.shape[0])
            damage[damage_type] = apply_normalization(d_data, stats)
            damage_parameter[damage_type] = d_param
            logger.info("Damage dataset '%s': %d instances", damage_type, d_data.shape[0])

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
        logger.info("Preprocessing completed and saved to %s", npz_path)

        return PreprocessedData(
            train=train, val=val, test=test, damage=damage, damage_parameter=damage_parameter, norm_stats=stats
        )
    except Exception:
        logger.exception("Preprocessing failed for the dataset in %s", dataset_paths.healthy.parent)
        raise
