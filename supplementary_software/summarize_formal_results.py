"""Create an auditable numerical digest for manuscript writing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRIMARY_SHOTS = [1, 2, 4]


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.aggregate_dir / "summary_family_bootstrap.csv")
    family = pd.read_csv(args.aggregate_dir / "family_metrics.csv")
    inference = pd.read_csv(args.aggregate_dir / "inference.csv")
    negative = pd.read_csv(args.aggregate_dir / "negative_transfer.csv")
    scalar = pd.read_csv(args.aggregate_dir / "scalar_predictions.csv.gz")

    selected_methods = [
        "explicit_ridge",
        "explicit_extra_trees",
        "weights_plus_handcrafted",
        "handcrafted_ridge",
        "random_fusion",
        "masked_only_fusion",
        "periodic_only_fusion",
        "topology_only_fusion",
        "ptopo_fusion",
        "ptopo_residual_ungated",
        "ptopo_star",
    ]
    headline = summary[
        summary["method"].isin(selected_methods) & summary["shots"].isin([0, 1, 2, 4])
    ].sort_values(["task", "shots", "macro_nrmse"])

    primary_family = family[
        family["shots"].isin(PRIMARY_SHOTS) & family["method"].isin(selected_methods)
    ]
    pooled = (
        primary_family.groupby(["task", "method"], as_index=False)
        .agg(pooled_family_macro_nrmse=("nrmse_global_sd", "mean"))
        .sort_values(["task", "pooled_family_macro_nrmse"])
    )

    energy = pooled[pooled["task"] == "energy_dissipation"].set_index("method")
    explicit_energy = float(energy.loc["explicit_ridge", "pooled_family_macro_nrmse"])
    star_energy = float(energy.loc["ptopo_star", "pooled_family_macro_nrmse"])
    masked_energy = float(energy.loc["masked_only_fusion", "pooled_family_macro_nrmse"])
    ptopo_energy = float(energy.loc["ptopo_fusion", "pooled_family_macro_nrmse"])
    ungated_energy = float(
        energy.loc["ptopo_residual_ungated", "pooled_family_macro_nrmse"]
    )
    relative_reduction = 100.0 * (explicit_energy - star_energy) / explicit_energy

    energy_family = primary_family[primary_family["task"] == "energy_dissipation"]
    star_family = (
        energy_family[energy_family["method"] == "ptopo_star"]
        .groupby(["held_out_family", "held_out_name"], as_index=False)
        .agg(star_nrmse=("nrmse_global_sd", "mean"))
    )
    explicit_family = (
        energy_family[energy_family["method"] == "explicit_ridge"]
        .groupby(["held_out_family", "held_out_name"], as_index=False)
        .agg(explicit_nrmse=("nrmse_global_sd", "mean"))
    )
    family_effects = star_family.merge(explicit_family, on=["held_out_family", "held_out_name"])
    family_effects["delta_nrmse"] = family_effects["star_nrmse"] - family_effects["explicit_nrmse"]
    family_effects = family_effects.sort_values("held_out_family")

    negative_pooled = (
        negative[negative["shots"].isin(PRIMARY_SHOTS)]
        .groupby(["task", "method"], as_index=False)
        .agg(negative_transfer_rate=("negative_transfer_rate", "mean"))
        .sort_values(["task", "negative_transfer_rate"])
    )
    energy_negative = negative_pooled[negative_pooled["task"] == "energy_dissipation"].set_index(
        "method"
    )
    star_negative = float(energy_negative.loc["ptopo_star", "negative_transfer_rate"])
    ungated_negative = float(
        energy_negative.loc["ptopo_residual_ungated", "negative_transfer_rate"]
    )

    frozen_inference = inference[
        inference["comparison_family"].isin(["performance", "mechanism", "attenuation"])
    ].set_index("comparison_family")
    periodic_topology_supported = bool(frozen_inference.loc["mechanism", "ci95_high"] < 0)
    star_performance_supported = bool(frozen_inference.loc["performance", "ci95_high"] < 0)
    gate_reduces_negative_transfer = bool(star_negative < ungated_negative)
    gate_macro_within_one_percent = bool(star_energy <= 1.01 * ungated_energy)
    gate_control_supported = gate_reduces_negative_transfer and gate_macro_within_one_percent
    if periodic_topology_supported and star_performance_supported:
        recommended_title = (
            "Periodic-topology self-supervision enables few-shot transfer across unseen "
            "dominant TPMS families"
        )
    else:
        recommended_title = (
            "A strict geometry-unseen evaluation of periodic-topology self-supervision "
            "for few-shot TPMS transfer"
        )

    star_scalar = scalar[(scalar["method"] == "ptopo_star") & scalar["shots"].isin(PRIMARY_SHOTS)]
    sample_gate = star_scalar.groupby(
        ["task", "held_out_family", "held_out_name", "shots", "sample_id"],
        as_index=False,
    ).agg(
        gate=("gate", "mean"),
        distance_norm=("distance_norm", "mean"),
        uncertainty_norm=("uncertainty_norm", "mean"),
    )
    gate_summary = (
        sample_gate.groupby("task", as_index=False)
        .agg(
            median_gate=("gate", "median"),
            mean_gate=("gate", "mean"),
            fraction_gate_lt_0_9=("gate", lambda x: float(np.mean(x < 0.9))),
            fraction_gate_lt_0_5=("gate", lambda x: float(np.mean(x < 0.5))),
            median_distance=("distance_norm", "median"),
            p90_distance=("distance_norm", lambda x: float(np.quantile(x, 0.9))),
            median_uncertainty=("uncertainty_norm", "median"),
        )
        .sort_values("task")
    )

    gate_rows: list[dict[str, Any]] = []
    for path in sorted(args.shard_dir.glob("fold*_seed*_gate_selections.json")):
        prefix = path.stem.replace("_gate_selections", "")
        fold_text, seed_text = prefix.split("_")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task, selection in payload.items():
            gate_rows.append(
                {
                    "fold": int(fold_text.removeprefix("fold")),
                    "seed": int(seed_text.removeprefix("seed")),
                    "task": task,
                    **selection,
                }
            )
    gate_selection = pd.DataFrame(gate_rows)
    gate_frequency = (
        gate_selection.groupby(
            ["task", "scale", "lambda_distance", "lambda_uncertainty"],
            as_index=False,
        )
        .agg(selections=("seed", "size"))
        .sort_values(["task", "selections"], ascending=[True, False])
    )

    digest = {
        "primary_shots": PRIMARY_SHOTS,
        "energy_pooled_explicit_nrmse": explicit_energy,
        "energy_pooled_ptopo_star_nrmse": star_energy,
        "energy_relative_reduction_percent": relative_reduction,
        "energy_pooled_masked_only_nrmse": masked_energy,
        "energy_pooled_ptopo_fusion_nrmse": ptopo_energy,
        "energy_pooled_ungated_nrmse": ungated_energy,
        "headline_family_bootstrap": records(headline),
        "pooled_family_macro": records(pooled),
        "primary_inference": records(inference),
        "energy_family_effects": records(family_effects),
        "energy_families_better": int((family_effects["delta_nrmse"] < 0).sum()),
        "energy_families_worse": int((family_effects["delta_nrmse"] > 0).sum()),
        "negative_transfer_pooled": records(negative_pooled),
        "gate_sample_summary": records(gate_summary),
        "gate_selection_frequency": records(gate_frequency),
        "claim_gates": {
            "periodic_topology_supported_by_paired_interval": periodic_topology_supported,
            "star_vs_explicit_supported_by_paired_interval": star_performance_supported,
            "gate_reduces_negative_transfer": gate_reduces_negative_transfer,
            "gate_macro_error_within_one_percent_of_ungated": gate_macro_within_one_percent,
            "controlled_negative_transfer_claim_supported": gate_control_supported,
            "star_negative_transfer_rate": star_negative,
            "ungated_negative_transfer_rate": ungated_negative,
            "recommended_title": recommended_title,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
