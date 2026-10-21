"""Aggregate complete shards and run physical-sample/family-level inference."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core import FullConfig, load_config
from scipy.stats import spearmanr


def scalar_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["task", "method", "shots", "held_out_family", "held_out_name", "seed", "repeat"]
    for values, frame in predictions.groupby(keys, sort=False):
        normalizer = float(frame["normalizer_sd"].iloc[0])
        correlation = spearmanr(frame["y_true"], frame["y_pred"]).statistic
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_test": len(frame),
                "nrmse_global_sd": float(np.sqrt(frame["squared_error"].mean()) / normalizer),
                "nmae_global_sd": float(frame["absolute_error"].mean() / normalizer),
                "spearman": float(correlation),
            }
        )
    return pd.DataFrame(rows)


def curve_run_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["task", "method", "shots", "held_out_family", "held_out_name", "seed", "repeat"]
    for values, frame in predictions.groupby(keys, sort=False):
        normalizer = float(frame["normalizer_sd"].iloc[0])
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_test": len(frame),
                "nrmse_global_sd": float(
                    np.sqrt(np.mean(frame["curve_rmse"].to_numpy() ** 2)) / normalizer
                ),
                "nmae_global_sd": float(frame["curve_mae"].mean() / normalizer),
                "energy_mae": float(frame["curve_energy_absolute_error"].mean()),
                "peak_mae": float(frame["curve_peak_absolute_error"].mean()),
                "spearman": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def family_summary(run_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = (
        run_metrics.groupby(
            ["task", "method", "shots", "held_out_family", "held_out_name"],
            as_index=False,
        )
        .agg(
            nrmse_global_sd=("nrmse_global_sd", "mean"),
            nmae_global_sd=("nmae_global_sd", "mean"),
            spearman=("spearman", "mean"),
            algorithm_runs=("nrmse_global_sd", "size"),
        )
    )
    summary = (
        family.groupby(["task", "method", "shots"], as_index=False)
        .agg(
            families=("held_out_family", "nunique"),
            macro_nrmse=("nrmse_global_sd", "mean"),
            between_family_nrmse_sd=("nrmse_global_sd", "std"),
            macro_nmae=("nmae_global_sd", "mean"),
            macro_spearman=("spearman", "mean"),
        )
        .sort_values(["task", "shots", "macro_nrmse"])
    )
    return family, summary


def family_bootstrap_summary(
    family_metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Compute descriptive intervals after averaging algorithmic repetitions.

    These unpaired intervals show variability across held-out families. Formal
    method comparisons use the paired family-plus-physical-sample bootstrap.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    keys = ["task", "method", "shots"]
    for values, frame in family_metrics.groupby(keys, sort=False):
        scores = frame["nrmse_global_sd"].to_numpy(dtype=np.float64)
        choices = rng.integers(0, len(scores), size=(replicates, len(scores)))
        draws = scores[choices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "families": len(scores),
                "macro_nrmse": float(scores.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "interval_type": "unpaired held-out-family bootstrap",
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "shots", "macro_nrmse"])


def paired_sample_table(
    predictions: pd.DataFrame,
    task: str,
    shots: list[int],
    method: str,
    reference: str,
    curve: bool = False,
) -> pd.DataFrame:
    subset = predictions.loc[
        (predictions["task"] == task)
        & predictions["shots"].isin(shots)
        & predictions["method"].isin([method, reference])
    ].copy()
    if subset.empty:
        raise ValueError(
            f"no paired predictions for task={task}, shots={shots}, "
            f"method={method}, reference={reference}"
        )
    keys = ["held_out_family", "sample_id", "seed", "shots", "repeat"]
    value = "curve_rmse" if curve else "squared_error"
    pivot = subset.pivot(index=keys, columns="method", values=value).dropna().reset_index()
    if curve:
        pivot[method] = pivot[method] ** 2
        pivot[reference] = pivot[reference] ** 2
    normalizer = float(subset["normalizer_sd"].iloc[0])
    collapsed = (
        pivot.groupby(["held_out_family", "sample_id"], as_index=False)
        .agg(method_mse=(method, "mean"), reference_mse=(reference, "mean"))
    )
    collapsed["normalizer"] = normalizer
    return collapsed


def family_effects(collapsed: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    for _, frame in collapsed.groupby("held_out_family"):
        normalizer = float(frame["normalizer"].iloc[0])
        values.append(
            (
                np.sqrt(frame["method_mse"].mean())
                - np.sqrt(frame["reference_mse"].mean())
            )
            / normalizer
        )
    return np.asarray(values)


def hierarchical_paired_bootstrap(
    collapsed: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    families = sorted(collapsed["held_out_family"].unique().tolist())
    frames = {family: collapsed[collapsed["held_out_family"] == family] for family in families}
    effects = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        family_differences: list[float] = []
        for family in sampled_families:
            frame = frames[int(family)]
            indices = rng.integers(0, len(frame), size=len(frame))
            sampled = frame.iloc[indices]
            normalizer = float(sampled["normalizer"].iloc[0])
            family_differences.append(
                (
                    np.sqrt(sampled["method_mse"].mean())
                    - np.sqrt(sampled["reference_mse"].mean())
                )
                / normalizer
            )
        effects[draw] = float(np.mean(family_differences))
    point = float(np.mean(family_effects(collapsed)))
    low, high = np.quantile(effects, [0.025, 0.975])
    return point, float(low), float(high)


def exact_sign_flip_pvalue(effects: np.ndarray) -> float:
    effects = np.asarray(effects, dtype=np.float64)
    observed = abs(float(effects.mean()))
    null = [
        abs(float(np.mean(effects * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(effects))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def negative_transfer_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "shots", "held_out_family", "seed", "repeat"]
    pivot = run_metrics.pivot(index=keys, columns="method", values="nrmse_global_sd")
    if "explicit_ridge" not in pivot:
        raise ValueError("explicit_ridge is required for negative-transfer analysis")
    rows: list[dict[str, Any]] = []
    for method in pivot.columns:
        if method == "explicit_ridge":
            continue
        valid = pivot[[method, "explicit_ridge"]].dropna()
        grouped = (valid[method] > valid["explicit_ridge"]).groupby(level=[0, 1, 2]).mean()
        for (task, shots, family), value in grouped.items():
            rows.append(
                {
                    "task": task,
                    "shots": shots,
                    "held_out_family": family,
                    "method": method,
                    "negative_transfer_rate": float(value),
                }
            )
    family = pd.DataFrame(rows)
    return (
        family.groupby(["task", "shots", "method"], as_index=False)
        .agg(
            families=("held_out_family", "nunique"),
            negative_transfer_rate=("negative_transfer_rate", "mean"),
            between_family_sd=("negative_transfer_rate", "std"),
        )
        .sort_values(["task", "shots", "negative_transfer_rate"])
    )


def validate_shards(shard_dir: Path, config: FullConfig) -> list[Path]:
    expected = {
        f"fold{fold}_seed{seed}_complete.json" for fold in range(8) for seed in config.seeds
    }
    present = {path.name for path in shard_dir.glob("*_complete.json")}
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"missing {len(missing)} downstream shards: {missing[:5]}")
    return sorted(shard_dir.glob("*_complete.json"))


def run_aggregation(config_path: Path, shard_dir: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    completion_files = validate_shards(shard_dir, config)
    scalar = pd.concat(
        [pd.read_csv(path.with_name(path.name.replace("_complete.json", "_scalar_predictions.csv.gz"))) for path in completion_files],
        ignore_index=True,
    )
    curve = pd.concat(
        [pd.read_csv(path.with_name(path.name.replace("_complete.json", "_curve_metrics.csv.gz"))) for path in completion_files],
        ignore_index=True,
    )
    scalar_metrics = scalar_run_metrics(scalar)
    curve_metrics = curve_run_metrics(curve)
    run_metrics = pd.concat([scalar_metrics, curve_metrics], ignore_index=True)
    family, summary = family_summary(run_metrics)
    summary_bootstrap = family_bootstrap_summary(
        family,
        config.bootstrap_replicates,
        seed=20260901,
    )
    negative = negative_transfer_summary(run_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar.to_csv(output_dir / "scalar_predictions.csv.gz", index=False, compression="gzip")
    curve.to_csv(output_dir / "curve_metrics.csv.gz", index=False, compression="gzip")
    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    family.to_csv(output_dir / "family_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    summary_bootstrap.to_csv(
        output_dir / "summary_family_bootstrap.csv",
        index=False,
    )
    negative.to_csv(output_dir / "negative_transfer.csv", index=False)

    comparison_specs = [
        ("performance", "ptopo_star", "explicit_ridge"),
        ("mechanism", "ptopo_fusion", "masked_only_fusion"),
        ("attenuation", "ptopo_star", "ptopo_residual_ungated"),
    ]
    inference_rows: list[dict[str, Any]] = []
    primary_pvalues: list[float] = []
    primary_indices: list[int] = []
    primary_shots = [shot for shot in (1, 2, 4) if shot in config.few_shots]
    if not primary_shots:
        raise ValueError("at least one of the frozen primary shots 1/2/4 is required")
    for comparison_index, (family_name, method, reference) in enumerate(comparison_specs):
        pooled = paired_sample_table(
            scalar,
            "energy_dissipation",
            primary_shots,
            method,
            reference,
        )
        effects = family_effects(pooled)
        point, low, high = hierarchical_paired_bootstrap(
            pooled,
            config.bootstrap_replicates,
            seed=20260901 + comparison_index,
        )
        pvalue = exact_sign_flip_pvalue(effects)
        primary_indices.append(len(inference_rows))
        primary_pvalues.append(pvalue)
        inference_rows.append(
            {
                "comparison_family": family_name,
                "task": "energy_dissipation",
                "shots": "|".join(map(str, primary_shots)),
                "method": method,
                "reference": reference,
                "effect_nrmse_method_minus_reference": point,
                "ci95_low": low,
                "ci95_high": high,
                "family_exact_sign_flip_p": pvalue,
                "holm_p": np.nan,
                "families": len(effects),
                "physical_samples": pooled["sample_id"].nunique(),
            }
        )
        for shots in primary_shots:
            collapsed = paired_sample_table(
                scalar,
                "energy_dissipation",
                [shots],
                method,
                reference,
            )
            shot_point, shot_low, shot_high = hierarchical_paired_bootstrap(
                collapsed,
                config.bootstrap_replicates,
                seed=20261001 + comparison_index * 10 + shots,
            )
            inference_rows.append(
                {
                    "comparison_family": f"{family_name}_descriptive",
                    "task": "energy_dissipation",
                    "shots": str(shots),
                    "method": method,
                    "reference": reference,
                    "effect_nrmse_method_minus_reference": shot_point,
                    "ci95_low": shot_low,
                    "ci95_high": shot_high,
                    "family_exact_sign_flip_p": np.nan,
                    "holm_p": np.nan,
                    "families": collapsed["held_out_family"].nunique(),
                    "physical_samples": collapsed["sample_id"].nunique(),
                }
            )
    adjusted = holm_adjust(primary_pvalues)
    for index, pvalue in zip(primary_indices, adjusted, strict=True):
        inference_rows[index]["holm_p"] = pvalue
    inference = pd.DataFrame(inference_rows)
    inference.to_csv(output_dir / "inference.csv", index=False)
    manifest = {
        "complete_shards": len(completion_files),
        "expected_shards": 8 * len(config.seeds),
        "scalar_prediction_rows": len(scalar),
        "curve_metric_rows": len(curve),
        "bootstrap_replicates": config.bootstrap_replicates,
        "inference_unit": "held-out topology family with physical-sample resampling",
        "algorithm_repetitions": {
            "pretraining_seeds": len(config.seeds),
            "support_repeats": config.support_repeats,
            "note": "algorithmic repetitions are averaged within physical samples and not treated as independent experiments",
        },
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path(__file__).with_name("results_full") / "downstream_shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_full") / "aggregate",
    )
    args = parser.parse_args()
    run_aggregation(args.config.resolve(), args.shard_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
