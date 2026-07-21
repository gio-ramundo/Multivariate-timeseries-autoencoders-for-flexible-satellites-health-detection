"""Discovery and loading of multi-damage combo datasets: one healthy file plus,
for every combination of LOW/MEDIUM/HIGH severity on stiffness/torque/gyroscope
(27 combinations), a file holding instances where all three damages are present
simultaneously, each drawn only from the parameter range of its combination's
severity level.

This is a different dataset shape than the single-damage-at-a-time datasets
handled by `preprocessing.py` (one damage type active per file, one damage
parameter per instance): here every instance carries all three damage
parameters at once, and the file itself fixes each parameter's severity level.
Feature selection and normalization are reused as-is from `preprocessing.py`
(imported, not modified) so a combo dataset is evaluated with the exact same
preprocessing the model was trained with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import NormalizationStats, apply_normalization, reorder_to_instance_first, select_features
from .utils.io import load_mat_v73
from .utils.logging_utils import get_logger

logger = get_logger(__name__)

# Order in which severity letters appear in a combo filename (e.g. "LMH" ->
# stiffness=LOW, torque=MEDIUM, gyroscope=HIGH), confirmed against the
# `severity_combo` array stored in each combo .mat file.
COMBO_DAMAGE_ORDER: tuple[str, ...] = ("stiffness", "torque", "gyroscope")
LEVEL_CODE_TO_INT: dict[str, int] = {"L": 1, "M": 2, "H": 3}
LEVEL_NAMES: dict[int, str] = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}

_HEALTHY_KEY = "dataset_healthy"
_COMBO_KEY = "dataset_combo"
_COMBO_NAME_RE = re.compile(r"_([LMH]{3})\.mat$", re.IGNORECASE)
_HEALTHY_NAME_RE = re.compile(r"_healthy\.mat$", re.IGNORECASE)


@dataclass
class ComboFile:
    combo: str  # 3-letter code, e.g. "LLL", in COMBO_DAMAGE_ORDER order
    path: Path
    levels: dict[str, int]  # damage_type -> 1/2/3


@dataclass
class ComboDataset:
    healthy_path: Path
    combos: list[ComboFile]


def discover_combo_dataset(data_dir: Path) -> ComboDataset:
    """Scans `data_dir` for exactly one `*_healthy.mat` file and any number of
    `*_<LMH><LMH><LMH>.mat` combo files (severity letters in
    COMBO_DAMAGE_ORDER order), independently of the filename prefix (e.g. the
    "300" in "300_healthy.mat" / "300_LLL.mat")."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Combo data folder not found: {data_dir}")

    files = [p for p in data_dir.iterdir() if p.is_file()]
    healthy_matches = [p for p in files if _HEALTHY_NAME_RE.search(p.name)]
    if len(healthy_matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*_healthy.mat' file in {data_dir}, found {len(healthy_matches)}: "
            f"{[p.name for p in healthy_matches]}"
        )

    combos: list[ComboFile] = []
    for p in sorted(files):
        m = _COMBO_NAME_RE.search(p.name)
        if not m:
            continue
        combo = m.group(1).upper()
        levels = {dt: LEVEL_CODE_TO_INT[code] for dt, code in zip(COMBO_DAMAGE_ORDER, combo)}
        combos.append(ComboFile(combo=combo, path=p, levels=levels))

    if not combos:
        raise FileNotFoundError(f"No '*_[LMH][LMH][LMH].mat' combo files found in {data_dir}")

    logger.info(
        "Discovered combo dataset in %s: healthy=%s, %d combo files (%s)",
        data_dir, healthy_matches[0].name, len(combos), ", ".join(sorted(c.combo for c in combos)),
    )
    return ComboDataset(healthy_path=healthy_matches[0], combos=combos)


@dataclass
class LoadedCombo:
    healthy: np.ndarray  # (n_instances, timesteps, n_features), normalized
    combo_data: dict[str, np.ndarray]  # combo code -> normalized array
    combo_params: dict[str, dict[str, np.ndarray]]  # combo code -> damage_type -> per-instance continuous param
    combo_levels: dict[str, dict[str, int]]  # combo code -> damage_type -> level (1/2/3)


def load_combo_dataset(dataset: ComboDataset, norm_stats: NormalizationStats) -> LoadedCombo:
    """Loads the healthy + combo .mat files and applies the given normalization
    stats, without refitting them: reusing exactly the mean/std the pretrained
    model was trained with is what makes its inference valid on this new
    dataset."""
    healthy_raw = load_mat_v73(dataset.healthy_path, [_HEALTHY_KEY])[_HEALTHY_KEY]
    if healthy_raw.ndim != 3:
        raise ValueError(f"'{_HEALTHY_KEY}' in {dataset.healthy_path} must have 3 dimensions, found {healthy_raw.ndim}")
    healthy = apply_normalization(select_features(reorder_to_instance_first(healthy_raw)), norm_stats)
    logger.info("Loaded healthy combo-analysis dataset: %d instances", healthy.shape[0])

    combo_data: dict[str, np.ndarray] = {}
    combo_params: dict[str, dict[str, np.ndarray]] = {}
    combo_levels: dict[str, dict[str, int]] = {}
    for cf in dataset.combos:
        raw = load_mat_v73(cf.path, [_COMBO_KEY, *COMBO_DAMAGE_ORDER])
        if raw[_COMBO_KEY].ndim != 3:
            raise ValueError(f"'{_COMBO_KEY}' in {cf.path} must have 3 dimensions, found {raw[_COMBO_KEY].ndim}")
        data = apply_normalization(select_features(reorder_to_instance_first(raw[_COMBO_KEY])), norm_stats)
        n_instances = data.shape[0]

        params: dict[str, np.ndarray] = {}
        for dt in COMBO_DAMAGE_ORDER:
            arr = np.asarray(raw[dt]).squeeze()
            if arr.ndim != 1 or arr.shape[0] != n_instances:
                raise ValueError(
                    f"Damage parameter '{dt}' in {cf.path} has unexpected shape {arr.shape} for {n_instances} instances"
                )
            params[dt] = arr

        combo_data[cf.combo] = data
        combo_params[cf.combo] = params
        combo_levels[cf.combo] = cf.levels
        logger.info("Loaded combo '%s' from %s: %d instances", cf.combo, cf.path.name, n_instances)

    return LoadedCombo(healthy=healthy, combo_data=combo_data, combo_params=combo_params, combo_levels=combo_levels)
