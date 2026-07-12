"""Ottimizzazione bayesiana (Optuna, multi-obiettivo) degli iperparametri.

Si minimizzano contemporaneamente l'errore di ricostruzione (MSE su validation
healthy) e il numero di parametri del modello. Usare la sola MSE come
obiettivo spingerebbe l'ottimizzatore verso gli iperparametri di massima
capacita' del range (piu' filtri/hidden units/latent_dim = errore piu' basso,
quasi sempre), producendo un autoencoder che tende all'identita' e quindi
inutile per il detection. La selezione finale dei trial da cui derivare i
range ristretti applica una regola di parsimonia: tra i trial con MSE entro
`tolerance` dal migliore, si scelgono i `top_n` a minore complessita'.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import HyperparamRange, get_search_space
from .models import build_model, count_parameters
from .preprocessing import PreprocessedData
from .utils.io import ResultsPaths, save_json, save_table
from .utils.logging_utils import get_logger


def _suggest_hyperparams(trial: optuna.Trial, search_space: dict[str, HyperparamRange]) -> dict[str, Any]:
    hp: dict[str, Any] = {}
    for name, rng in search_space.items():
        if rng.kind == "categorical":
            hp[name] = trial.suggest_categorical(name, rng.choices)
        elif rng.kind == "int":
            hp[name] = trial.suggest_int(name, int(rng.low), int(rng.high), log=rng.log)
        else:
            hp[name] = trial.suggest_float(name, rng.low, rng.high, log=rng.log)
    # padding dipende dal kernel_size campionato in questo stesso trial.
    hp["padding"] = trial.suggest_int("padding", 0, hp["kernel_size"] // 2)
    return hp


def build_objective(
    arch_name: str,
    data: PreprocessedData,
    input_len: int,
    n_features: int,
    hpo_epochs: int,
    device: torch.device,
    logger,
    seed: int = 0,
):
    search_space = get_search_space(arch_name)
    val_tensor = torch.from_numpy(data.val).float().to(device)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        try:
            torch.manual_seed(seed + trial.number)
            hp = _suggest_hyperparams(trial, search_space)
            model = build_model(arch_name, hp, input_len, n_features).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
            loss_fn = nn.MSELoss()

            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(data.train).float()),
                batch_size=hp["batch_size"],
                shuffle=True,
            )

            model.train()
            for _ in range(hpo_epochs):
                for (batch,) in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon = model(batch)
                    loss = loss_fn(recon, batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                val_recon = model(val_tensor)
                val_mse = loss_fn(val_recon, val_tensor).item()

            if not np.isfinite(val_mse):
                raise ValueError(f"val_mse non finito: {val_mse}")

            return val_mse, float(count_parameters(model))
        except (ValueError, RuntimeError) as exc:
            logger.warning("Trial %d scartato (combinazione non valida): %s", trial.number, exc)
            raise optuna.TrialPruned() from exc
        except Exception:
            logger.exception("Trial %d fallito per errore inatteso", trial.number)
            raise optuna.TrialPruned()

    return objective


def select_parsimonious_trials(
    study: optuna.Study, top_n: int, tolerance: float = 0.05
) -> list[optuna.trial.FrozenTrial]:
    """Tra i trial completati entro `tolerance` relativa dal miglior val_mse,
    seleziona i `top_n` a minor numero di parametri (regola di parsimonia)."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not completed:
        raise RuntimeError("Nessun trial completato: impossibile derivare i range ristretti")

    best_mse = min(t.values[0] for t in completed)
    threshold = best_mse * (1.0 + tolerance)
    candidates = [t for t in completed if t.values[0] <= threshold]
    candidates.sort(key=lambda t: t.values[1])  # complessita' crescente
    return candidates[:top_n]


def compute_narrow_ranges(
    trials: list[optuna.trial.FrozenTrial], search_space: dict[str, HyperparamRange], margin: float = 0.1
) -> dict[str, HyperparamRange]:
    param_names = list(search_space.keys()) + ["padding"]
    narrowed: dict[str, HyperparamRange] = {}

    for name in param_names:
        values = [t.params[name] for t in trials]
        base = search_space.get(name)

        if name == "padding" or (base is not None and base.kind == "categorical"):
            uniques = sorted(set(values))
            if name == "padding":
                narrowed[name] = HyperparamRange(kind="int", low=min(uniques), high=max(uniques))
            else:
                narrowed[name] = HyperparamRange(kind="categorical", choices=uniques)
        else:
            lo, hi = float(min(values)), float(max(values))
            span = hi - lo
            pad = span * margin if span > 0 else max(abs(lo) * margin, 1e-8)
            new_lo = max(lo - pad, base.low)
            new_hi = min(hi + pad, base.high)
            narrowed[name] = HyperparamRange(kind=base.kind, low=new_lo, high=new_hi, log=base.log)

    return narrowed


def run_bayesian_optimization(
    arch_name: str,
    data: PreprocessedData,
    results_paths: ResultsPaths,
    n_trials: int,
    top_n: int,
    hpo_epochs: int,
    tolerance: float,
    seed: int = 0,
) -> dict[str, HyperparamRange]:
    logger = get_logger("bayesian_optimizer", results_paths.logs / "bayesian_optimizer.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selezionato per l'HPO: %s", device)

    input_len = data.train.shape[1]
    n_features = data.train.shape[2]

    storage_path = results_paths.hpo / "optuna_study.db"
    storage_url = f"sqlite:///{storage_path}"

    try:
        study = optuna.create_study(
            study_name="hpo",
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=seed),
            storage=storage_url,
            load_if_exists=True,
        )

        n_already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, n_trials - n_already_done)
        logger.info("Trial completati in precedenza: %d, da eseguire ora: %d", n_already_done, remaining)

        objective = build_objective(arch_name, data, input_len, n_features, hpo_epochs, device, logger, seed=seed)

        start_time = time.time()
        if remaining > 0:
            study.optimize(objective, n_trials=remaining, catch=())
        elapsed = time.time() - start_time

        selected = select_parsimonious_trials(study, top_n=top_n, tolerance=tolerance)
        logger.info("Selezionati %d trial parsimoniosi su %d completati", len(selected), n_already_done + remaining)

        search_space = get_search_space(arch_name)
        narrowed_ranges = compute_narrow_ranges(selected, search_space)

        save_json({k: asdict(v) for k, v in narrowed_ranges.items()}, results_paths.hpo / "narrowed_ranges.json")

        trials_df = study.trials_dataframe()
        save_table(trials_df, results_paths.hpo / "all_trials", formats=("csv",))

        n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        save_json(
            {
                "n_trials_run_this_call": remaining,
                "elapsed_seconds_this_call": elapsed,
                "mean_seconds_per_trial": elapsed / remaining if remaining else None,
                "n_completed_trials_total": n_completed_trials,
            },
            results_paths.hpo / "execution_times.json",
        )

        _plot_pareto_front(study, selected, results_paths.hpo / "pareto_front.pdf")

        return narrowed_ranges
    except Exception:
        logger.exception("Ottimizzazione bayesiana fallita per architettura '%s'", arch_name)
        raise


def _plot_pareto_front(study: optuna.Study, selected: list[optuna.trial.FrozenTrial], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        selected_numbers = {t.number for t in selected}

        fig, ax = plt.subplots(figsize=(6, 5))
        xs_all = [t.values[0] for t in completed]
        ys_all = [t.values[1] for t in completed]
        ax.scatter(xs_all, ys_all, c="tab:gray", alpha=0.5, label="tutti i trial")

        xs_sel = [t.values[0] for t in completed if t.number in selected_numbers]
        ys_sel = [t.values[1] for t in completed if t.number in selected_numbers]
        ax.scatter(xs_sel, ys_sel, c="tab:red", label="selezionati (parsimoniosi)")

        ax.set_xlabel("val MSE")
        ax.set_ylabel("n. parametri")
        ax.set_title("Fronte errore/complessita'")
        ax.legend()
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        get_logger("bayesian_optimizer").exception("Impossibile generare il grafico del fronte di Pareto")
