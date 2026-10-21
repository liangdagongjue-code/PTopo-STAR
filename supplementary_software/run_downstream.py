"""Run one fold/seed shard of the PTopo-STAR few-shot evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core import PRIMITIVE_NAMES, build_geometry_unseen_cache, load_config
from downstream import (
    fit_predict_ridge,
    fit_residual_system,
    gate_values,
    select_gate_source_only,
)
from sklearn.ensemble import ExtraTreesRegressor


def load_embedding(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        ids = data["target_ids"]
        embeddings = data["embeddings"]
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"target sample order mismatch in {path}")
    return embeddings.astype(np.float64)


def support_indices(
    target: np.ndarray,
    shots: int,
    repeat: int,
    held_out_family: int,
    support_seed: int,
) -> np.ndarray:
    if shots == 0:
        return np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(
        support_seed + 100003 * held_out_family + 1009 * shots + repeat
    )
    return np.sort(rng.choice(target, size=shots, replace=False))


def curve_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    global_stress_sd: float,
) -> dict[str, np.ndarray]:
    error = y_pred - y_true
    return {
        "rmse": np.sqrt(np.mean(error**2, axis=1)),
        "nrmse_global_sd": np.sqrt(np.mean(error**2, axis=1)) / global_stress_sd,
        "mae": np.mean(np.abs(error), axis=1),
    }


def curve_energy(curves: np.ndarray, strain: np.ndarray) -> np.ndarray:
    half = curves.shape[1] // 2
    dx = np.max(strain, axis=1) / half
    difference = curves[:, :half] - curves[:, half:][:, ::-1]
    return np.trapezoid(difference, dx=dx[:, None], axis=1)


def append_scalar_rows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: np.ndarray,
    normalizer: float,
    gate: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> None:
    n = len(y_true)
    gate = np.full(n, np.nan) if gate is None else gate
    distance = np.full(n, np.nan) if distance is None else distance
    uncertainty = np.full(n, np.nan) if uncertainty is None else uncertainty
    for index in range(n):
        error = float(y_pred[index] - y_true[index])
        rows.append(
            {
                **metadata,
                "method": method,
                "sample_id": str(sample_ids[index]),
                "y_true": float(y_true[index]),
                "y_pred": float(y_pred[index]),
                "error": error,
                "absolute_error": abs(error),
                "squared_error": error**2,
                "normalizer_sd": normalizer,
                "gate": float(gate[index]),
                "distance_norm": float(distance[index]),
                "uncertainty_norm": float(uncertainty[index]),
            }
        )


def append_curve_rows(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strain: np.ndarray,
    official_energy: np.ndarray,
    sample_ids: np.ndarray,
    global_stress_sd: float,
    gate: np.ndarray | None = None,
    distance: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
) -> None:
    metrics = curve_metrics(y_true, y_pred, global_stress_sd)
    predicted_energy = curve_energy(y_pred, strain)
    predicted_peak = np.max(y_pred, axis=1)
    true_peak = np.max(y_true, axis=1)
    n = len(y_true)
    gate = np.full(n, np.nan) if gate is None else gate
    distance = np.full(n, np.nan) if distance is None else distance
    uncertainty = np.full(n, np.nan) if uncertainty is None else uncertainty
    for index in range(n):
        rows.append(
            {
                **metadata,
                "method": method,
                "sample_id": str(sample_ids[index]),
                "curve_rmse": float(metrics["rmse"][index]),
                "curve_nrmse_global_sd": float(metrics["nrmse_global_sd"][index]),
                "curve_mae": float(metrics["mae"][index]),
                "official_energy": float(official_energy[index]),
                "predicted_curve_energy": float(predicted_energy[index]),
                "curve_energy_absolute_error": float(
                    abs(predicted_energy[index] - official_energy[index])
                ),
                "true_peak": float(true_peak[index]),
                "predicted_curve_peak": float(predicted_peak[index]),
                "curve_peak_absolute_error": float(abs(predicted_peak[index] - true_peak[index])),
                "normalizer_sd": global_stress_sd,
                "gate": float(gate[index]),
                "distance_norm": float(distance[index]),
                "uncertainty_norm": float(uncertainty[index]),
            }
        )


def run_shard(
    config_path: Path,
    pretraining_dir: Path,
    output_dir: Path,
    held_out_family: int,
    seed: int,
) -> None:
    config = load_config(config_path)
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not listed in {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_stem = f"fold{held_out_family}_seed{seed}"
    completion_path = output_dir / f"{shard_stem}_complete.json"
    if completion_path.exists():
        print(json.dumps({"status": "resume_skip", "shard": shard_stem}))
        return
    run_root = Path(__file__).resolve().parents[2]
    payload = build_geometry_unseen_cache(config, run_root, held_out_family, seed)
    target_ids = payload["target_ids"]
    weights = payload["target_weights"].astype(np.float64)
    descriptors = payload["target_descriptors"].astype(np.float64)
    groups = payload["target_groups"]
    energy = payload["target_energy"].astype(np.float64)
    peak = payload["target_peak"].astype(np.float64)
    curves = payload["target_curve"].astype(np.float64)
    strains = payload["target_strain"].astype(np.float64)
    fold_dir = pretraining_dir / f"fold_{held_out_family}" / f"seed_{seed}"
    embeddings = {
        "random_encoder": load_embedding(
            fold_dir / "target_embeddings_random_encoder.npz", target_ids
        )
    }
    for variant in config.variants:
        embeddings[variant] = load_embedding(
            fold_dir / f"target_embeddings_{variant}.npz", target_ids
        )

    source = np.flatnonzero(groups != held_out_family)
    target = np.flatnonzero(groups == held_out_family)
    if len(target) < max(config.few_shots) + config.minimum_test_samples:
        raise ValueError(
            f"held-out family {held_out_family} has only {len(target)} samples, "
            "insufficient for the configured maximum shot"
        )
    tasks = {
        "energy_dissipation": energy,
        "peak_stress": peak,
        "stress_curve": curves,
    }
    gate_selections: dict[str, Any] = {}
    for task_index, (task_name, y) in enumerate(tasks.items()):
        explicit_features = (
            np.concatenate([weights, strains], axis=1)
            if task_name == "stress_curve"
            else weights
        )
        selection, _ = select_gate_source_only(
            explicit_features[source],
            embeddings["ptopo_ssl"][source],
            y[source],
            groups[source],
            config.ridge_alphas,
            config.inner_residual_ensemble_size,
            seed + 1000003 * held_out_family + 10007 * task_index,
        )
        gate_selections[task_name] = selection

    scalar_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    curve_prediction_rows: list[np.ndarray] = []
    curve_truth_rows: list[np.ndarray] = []
    curve_strain_rows: list[np.ndarray] = []
    curve_prediction_metadata: list[tuple[Any, ...]] = []
    split_rows: list[dict[str, Any]] = []
    direct_features = {
        "explicit_ridge": weights,
        "handcrafted_ridge": descriptors,
        "weights_plus_handcrafted": np.concatenate([weights, descriptors], axis=1),
        "random_fusion": np.concatenate([weights, embeddings["random_encoder"]], axis=1),
    }
    for variant in config.variants:
        method_name = "ptopo_fusion" if variant == "ptopo_ssl" else f"{variant}_fusion"
        direct_features[method_name] = np.concatenate(
            [weights, embeddings[variant]], axis=1
        )
    full_curve_methods = {
        "explicit_ridge",
        "explicit_extra_trees",
        "masked_only_fusion",
        "ptopo_fusion",
        "ptopo_residual_ungated",
        "ptopo_star",
    }
    global_normalizers = {
        "energy_dissipation": float(np.std(energy, ddof=1)),
        "peak_stress": float(np.std(peak, ddof=1)),
        "stress_curve": float(np.std(curves, ddof=1)),
    }
    for shots in config.few_shots:
        repeats = 1 if shots == 0 else config.support_repeats
        for repeat in range(repeats):
            support = support_indices(
                target,
                shots,
                repeat,
                held_out_family,
                config.support_seed,
            )
            test = np.setdiff1d(target, support)
            train = np.concatenate([source, support])
            scalar_tree_model = ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=1.0,
                n_jobs=-1,
                random_state=(
                    seed + 1000003 * held_out_family + 101 * shots + repeat
                ),
            )
            scalar_tree_model.fit(
                weights[train], np.column_stack([energy, peak])[train]
            )
            scalar_tree_prediction = scalar_tree_model.predict(weights[test])
            curve_explicit_features = np.concatenate([weights, strains], axis=1)
            curve_tree_model = ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                max_features=1.0,
                n_jobs=-1,
                random_state=(
                    seed + 1000003 * held_out_family + 101 * shots + repeat + 17
                ),
            )
            curve_tree_model.fit(curve_explicit_features[train], curves[train])
            curve_tree_prediction = curve_tree_model.predict(
                curve_explicit_features[test]
            )
            tree_predictions = {
                "energy_dissipation": scalar_tree_prediction[:, 0],
                "peak_stress": scalar_tree_prediction[:, 1],
                "stress_curve": curve_tree_prediction,
            }
            split_rows.append(
                {
                    "held_out_family": held_out_family,
                    "held_out_name": PRIMITIVE_NAMES[held_out_family],
                    "shots": shots,
                    "repeat": repeat,
                    "support_ids": "|".join(target_ids[support].tolist()),
                    "test_ids": "|".join(target_ids[test].tolist()),
                    "n_source": len(source),
                    "n_support": len(support),
                    "n_test": len(test),
                }
            )
            for task_index, (task_name, y) in enumerate(tasks.items()):
                metadata = {
                    "seed": seed,
                    "held_out_family": held_out_family,
                    "held_out_name": PRIMITIVE_NAMES[held_out_family],
                    "task": task_name,
                    "shots": shots,
                    "repeat": repeat,
                    "n_source": len(source),
                    "n_support": len(support),
                    "n_test": len(test),
                }
                method_predictions: dict[str, np.ndarray] = {}
                for method, features in direct_features.items():
                    task_features = (
                        np.concatenate([features, strains], axis=1)
                        if task_name == "stress_curve"
                        else features
                    )
                    method_predictions[method] = fit_predict_ridge(
                        task_features[train],
                        y[train],
                        task_features[test],
                        config.ridge_alphas,
                    )
                method_predictions["explicit_extra_trees"] = tree_predictions[task_name]
                explicit_task_features = (
                    curve_explicit_features
                    if task_name == "stress_curve"
                    else weights
                )
                system = fit_residual_system(
                    explicit_task_features[train],
                    embeddings["ptopo_ssl"][train],
                    y[train],
                    groups[train],
                    explicit_task_features[test],
                    embeddings["ptopo_ssl"][test],
                    config.ridge_alphas,
                    config.residual_ensemble_size,
                    seed
                    + 1000003 * held_out_family
                    + 10007 * task_index
                    + 101 * shots
                    + repeat,
                )
                base_shape = np.asarray(system.base).shape
                ungated = np.asarray(system.base) + np.asarray(system.residual_mean)
                selection = gate_selections[task_name]
                gate = gate_values(
                    system.distance_norm,
                    system.uncertainty_norm,
                    selection.scale,
                    selection.lambda_distance,
                    selection.lambda_uncertainty,
                )
                gate_for_output = gate if len(base_shape) == 1 else gate[:, None]
                star = np.asarray(system.base) + gate_for_output * np.asarray(system.residual_mean)
                method_predictions["ptopo_residual_ungated"] = ungated
                method_predictions["ptopo_star"] = star

                if task_name != "stress_curve":
                    for method, prediction in method_predictions.items():
                        is_residual = method in {"ptopo_residual_ungated", "ptopo_star"}
                        append_scalar_rows(
                            scalar_rows,
                            metadata,
                            method,
                            y[test],
                            prediction,
                            target_ids[test],
                            global_normalizers[task_name],
                            gate=(
                                np.ones(len(test))
                                if method == "ptopo_residual_ungated"
                                else gate
                                if method == "ptopo_star"
                                else None
                            ),
                            distance=system.distance_norm if is_residual else None,
                            uncertainty=system.uncertainty_norm if is_residual else None,
                        )
                else:
                    for method, prediction in method_predictions.items():
                        is_residual = method in {"ptopo_residual_ungated", "ptopo_star"}
                        append_curve_rows(
                            curve_rows,
                            metadata,
                            method,
                            y[test],
                            prediction,
                            strains[test],
                            energy[test],
                            target_ids[test],
                            global_normalizers[task_name],
                            gate=(
                                np.ones(len(test))
                                if method == "ptopo_residual_ungated"
                                else gate
                                if method == "ptopo_star"
                                else None
                            ),
                            distance=system.distance_norm if is_residual else None,
                            uncertainty=system.uncertainty_norm if is_residual else None,
                        )
                        if repeat == 0 and method in full_curve_methods:
                            curve_prediction_rows.append(np.asarray(prediction, dtype=np.float32))
                            curve_truth_rows.append(np.asarray(y[test], dtype=np.float32))
                            curve_strain_rows.append(
                                np.asarray(strains[test], dtype=np.float32)
                            )
                            curve_prediction_metadata.extend(
                                [
                                    (
                                        seed,
                                        held_out_family,
                                        shots,
                                        repeat,
                                        method,
                                        str(sample_id),
                                    )
                                    for sample_id in target_ids[test]
                                ]
                            )

    pd.DataFrame(scalar_rows).to_csv(
        output_dir / f"{shard_stem}_scalar_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(curve_rows).to_csv(
        output_dir / f"{shard_stem}_curve_metrics.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(split_rows).to_csv(output_dir / f"{shard_stem}_splits.csv", index=False)
    if curve_prediction_rows:
        metadata_array = np.asarray(
            curve_prediction_metadata,
            dtype=[
                ("seed", "i8"),
                ("held_out_family", "i8"),
                ("shots", "i8"),
                ("repeat", "i8"),
                ("method", "U40"),
                ("sample_id", "U128"),
            ],
        )
        np.savez_compressed(
            output_dir / f"{shard_stem}_curve_predictions_repeat0.npz",
            metadata=metadata_array,
            predictions=np.concatenate(curve_prediction_rows, axis=0),
            truth=np.concatenate(curve_truth_rows, axis=0),
            strain=np.concatenate(curve_strain_rows, axis=0),
        )
    gate_json = {name: asdict(selection) for name, selection in gate_selections.items()}
    (output_dir / f"{shard_stem}_gate_selections.json").write_text(
        json.dumps(gate_json, indent=2),
        encoding="utf-8",
    )
    completion_path.write_text(
        json.dumps(
            {
                "held_out_family": held_out_family,
                "held_out_name": PRIMITIVE_NAMES[held_out_family],
                "seed": seed,
                "scalar_prediction_rows": len(scalar_rows),
                "curve_metric_rows": len(curve_rows),
                "split_rows": len(split_rows),
                "source_only_gate_selection": True,
                "completed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--pretraining-dir",
        type=Path,
        default=Path(__file__).with_name("results_full") / "pretraining",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_full") / "downstream_shards",
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(8))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run_shard(
        args.config.resolve(),
        args.pretraining_dir.resolve(),
        args.output_dir.resolve(),
        args.fold,
        args.seed,
    )


if __name__ == "__main__":
    main()
