"""Final training with checkpoint/resume, inference on healthy and the 3 damage
datasets, error computation (total and per-feature) and correlations with the
damage level.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model
from .preprocessing import PreprocessedData
from .utils.io import ResultsPaths, load_checkpoint, save_checkpoint, save_json, save_table
from .utils.logging_utils import get_logger


def train_final_model(
    arch_name: str,
    hp: dict[str, Any],
    data: PreprocessedData,
    results_paths: ResultsPaths,
    epochs: int,
    resume: bool = True,
    seed: int = 0,
) -> tuple[nn.Module, pd.DataFrame]:
    logger = get_logger("train_test", results_paths.logs / "train_test.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selected for final training: %s", device)

    input_len = data.train.shape[1]
    n_features = data.train.shape[2]

    torch.manual_seed(seed)
    model = build_model(arch_name, hp, input_len, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    checkpoint_path = results_paths.training / "checkpoint.pt"
    best_model_path = results_paths.training / "best_model.pt"
    curve_path = results_paths.training / "training_curve.csv"

    start_epoch = 0
    best_val_loss = float("inf")
    curve_records: list[dict[str, Any]] = []

    if resume and checkpoint_path.is_file():
        try:
            last_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, map_location=str(device))
            start_epoch = last_epoch + 1
            if curve_path.is_file():
                curve_records = pd.read_csv(curve_path).to_dict("records")
            logger.info("Training resumed from epoch %d (best_val_loss=%.6g)", start_epoch, best_val_loss)
        except Exception:
            logger.exception("Checkpoint could not be restored, starting from scratch")
            start_epoch, best_val_loss, curve_records = 0, float("inf"), []

    train_loader = DataLoader(TensorDataset(torch.from_numpy(data.train).float()), batch_size=hp["batch_size"], shuffle=True)
    val_tensor = torch.from_numpy(data.val).float().to(device)

    training_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        batch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensor), val_tensor).item()

        train_loss = float(np.mean(batch_losses))
        curve_records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("Epoch %d/%d - train_loss=%.6g val_loss=%.6g", epoch + 1, epochs, train_loss, val_loss)

        save_checkpoint(checkpoint_path, model, optimizer, epoch, min(best_val_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_model_path, model, optimizer, epoch, best_val_loss)

        pd.DataFrame(curve_records).to_csv(curve_path, index=False)

    training_elapsed = time.time() - training_start

    if best_model_path.is_file():
        load_checkpoint(best_model_path, model, optimizer, map_location=str(device))
    else:
        logger.warning("No best_model saved (0 epochs run?): using current weights")

    save_json(
        {"training_seconds": training_elapsed, "epochs_run": max(0, epochs - start_epoch), "best_val_loss": best_val_loss},
        results_paths.training / "training_time.json",
    )

    return model, pd.DataFrame(curve_records)


def run_inference(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 32) -> tuple[np.ndarray, float]:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x).float()), batch_size=batch_size, shuffle=False)

    outputs = []
    start = time.time()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            outputs.append(recon.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return np.concatenate(outputs, axis=0), elapsed


def compute_errors(x: np.ndarray, x_hat: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-instance errors: total (mse/mae/rmse) and per-feature."""
    diff = x - x_hat
    se = diff**2
    ae = np.abs(diff)

    mse_total = se.mean(axis=(1, 2))
    mae_total = ae.mean(axis=(1, 2))
    rmse_total = np.sqrt(mse_total)
    total_df = pd.DataFrame({"mse": mse_total, "mae": mae_total, "rmse": rmse_total})

    mse_feat = se.mean(axis=1)
    mae_feat = ae.mean(axis=1)
    rmse_feat = np.sqrt(mse_feat)
    n_features = x.shape[-1]
    per_feature_df = pd.DataFrame(
        {
            **{f"mse_f{i}": mse_feat[:, i] for i in range(n_features)},
            **{f"mae_f{i}": mae_feat[:, i] for i in range(n_features)},
            **{f"rmse_f{i}": rmse_feat[:, i] for i in range(n_features)},
        }
    )
    return total_df, per_feature_df


def compute_damage_correlations(error_values: np.ndarray, damage_parameter: np.ndarray) -> dict[str, float]:
    pearson_r, pearson_p = pearsonr(error_values, damage_parameter)
    tau, tau_p = kendalltau(error_values, damage_parameter, variant="b")
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "kendall_tau_b": float(tau),
        "kendall_p": float(tau_p),
    }


def run_train_test(
    arch_name: str,
    hp: dict[str, Any],
    data: PreprocessedData,
    results_paths: ResultsPaths,
    epochs: int,
    resume: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    logger = get_logger("train_test", results_paths.logs / "train_test.log")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, curve = train_final_model(arch_name, hp, data, results_paths, epochs, resume=resume, seed=seed)

        batch_size = hp["batch_size"]
        recon_healthy, t_healthy = run_inference(model, data.test, device, batch_size=batch_size)
        total_healthy, per_feature_healthy = compute_errors(data.test, recon_healthy)
        total_healthy["dataset"] = "healthy"
        total_healthy["damage_parameter"] = np.nan
        per_feature_healthy["dataset"] = "healthy"
        per_feature_healthy["damage_parameter"] = np.nan

        all_totals = [total_healthy]
        all_per_feature = [per_feature_healthy]
        predictions = {"healthy": recon_healthy}
        inference_times = {"healthy": t_healthy}
        correlations: dict[str, dict[str, Any]] = {}

        for damage_type, damage_data in data.damage.items():
            recon, t = run_inference(model, damage_data, device, batch_size=batch_size)
            total_df, per_feature_df = compute_errors(damage_data, recon)
            damage_parameter = data.damage_parameter[damage_type]

            total_df["dataset"] = damage_type
            total_df["damage_parameter"] = damage_parameter
            per_feature_df["dataset"] = damage_type
            per_feature_df["damage_parameter"] = damage_parameter

            all_totals.append(total_df)
            all_per_feature.append(per_feature_df)
            predictions[damage_type] = recon
            inference_times[damage_type] = t

            n_features = damage_data.shape[-1]
            correlations[damage_type] = {
                "total": {
                    metric: compute_damage_correlations(total_df[metric].to_numpy(), damage_parameter)
                    for metric in ("mse", "mae", "rmse")
                },
                "per_feature": {
                    metric: {
                        f"f{feature_idx}": compute_damage_correlations(
                            per_feature_df[f"{metric}_f{feature_idx}"].to_numpy(), damage_parameter
                        )
                        for feature_idx in range(n_features)
                    }
                    for metric in ("mse", "mae", "rmse")
                },
            }

        errors_df = pd.concat(all_totals, ignore_index=True)
        errors_per_feature_df = pd.concat(all_per_feature, ignore_index=True)

        save_table(errors_df, results_paths.evaluation / "errors", formats=("csv",))
        save_table(errors_per_feature_df, results_paths.evaluation / "errors_per_feature", formats=("csv",))
        save_json(correlations, results_paths.evaluation / "correlations.json")
        save_json(inference_times, results_paths.evaluation / "inference_times.json")

        np.savez_compressed(
            results_paths.evaluation / "predictions.npz", **{f"pred_{k}": v for k, v in predictions.items()}
        )

        logger.info("Train/test completed for '%s'", arch_name)
        return {
            "model": model,
            "training_curve": curve,
            "errors": errors_df,
            "errors_per_feature": errors_per_feature_df,
            "correlations": correlations,
            "inference_times": inference_times,
            "predictions": predictions,
        }
    except Exception:
        logger.exception("Train/test failed for architecture '%s'", arch_name)
        raise
