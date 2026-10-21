"""Fail-closed integrity audit for a completed PTopo-STAR formal run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pretraining(results_dir: Path, config: dict[str, object]) -> dict[str, int]:
    seeds = [int(seed) for seed in config["seeds"]]
    variants = [str(variant) for variant in config["variants"]]
    manifests = 0
    checkpoints = 0
    histories = 0
    for fold in range(8):
        for seed in seeds:
            fold_dir = results_dir / "pretraining" / f"fold_{fold}" / f"seed_{seed}"
            manifest_path = fold_dir / "pretraining_complete.json"
            if not manifest_path.exists():
                raise ValueError(f"missing pretraining manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["held_out_family"] != fold or manifest["seed"] != seed:
                raise ValueError(f"fold/seed mismatch in {manifest_path}")
            if set(manifest["completed_variants"]) != set(variants):
                raise ValueError(f"variant mismatch in {manifest_path}")
            if manifest["strict_geometry_unseen"] is not True:
                raise ValueError(f"strict holdout disabled in {manifest_path}")
            if float(manifest["maximum_held_out_pretraining_weight"]) != 0.0:
                raise ValueError(f"pretraining leakage in {manifest_path}")
            history_path = fold_dir / "training_history.csv"
            history = pd.read_csv(history_path)
            expected_history_rows = len(variants) * int(config["pretrain_epochs"])
            if len(history) != expected_history_rows:
                raise ValueError(f"history row mismatch in {history_path}")
            histories += 1
            for variant in variants:
                required = (
                    fold_dir / f"model_fold{fold}_seed{seed}_{variant}.pt",
                    fold_dir / f"target_embeddings_{variant}.npz",
                )
                if not all(path.exists() and path.stat().st_size > 0 for path in required):
                    raise ValueError(
                        f"missing checkpoint or embedding for fold={fold}, seed={seed}, variant={variant}"
                    )
                checkpoints += 1
            random_embedding = fold_dir / "target_embeddings_random_encoder.npz"
            if not random_embedding.exists() or random_embedding.stat().st_size == 0:
                raise ValueError(f"missing random embedding: {random_embedding}")
            manifests += 1
    return {"pretraining_manifests": manifests, "checkpoints": checkpoints, "histories": histories}


def audit_downstream(results_dir: Path, config: dict[str, object]) -> dict[str, int]:
    seeds = [int(seed) for seed in config["seeds"]]
    expected_splits = sum(
        1 if int(shots) == 0 else int(config["support_repeats"]) for shots in config["few_shots"]
    )
    shard_dir = results_dir / "downstream_shards"
    shards = 0
    split_rows = 0
    for fold in range(8):
        family_reference: pd.DataFrame | None = None
        for seed in seeds:
            stem = f"fold{fold}_seed{seed}"
            completion_path = shard_dir / f"{stem}_complete.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if not completion["completed"] or not completion["source_only_gate_selection"]:
                raise ValueError(f"invalid completion marker: {completion_path}")
            if completion["held_out_family"] != fold or completion["seed"] != seed:
                raise ValueError(f"fold/seed mismatch in {completion_path}")
            split_path = shard_dir / f"{stem}_splits.csv"
            splits = pd.read_csv(split_path).sort_values(["shots", "repeat"]).reset_index(drop=True)
            if len(splits) != expected_splits or int(completion["split_rows"]) != expected_splits:
                raise ValueError(f"split count mismatch in {split_path}")
            support_columns = ["shots", "repeat", "support_ids", "test_ids", "n_support", "n_test"]
            if family_reference is None:
                family_reference = splits[support_columns]
            elif not family_reference.equals(splits[support_columns]):
                raise ValueError(f"split identifiers differ across encoder seeds in fold {fold}")
            required_paths = (
                shard_dir / f"{stem}_scalar_predictions.csv.gz",
                shard_dir / f"{stem}_curve_metrics.csv.gz",
                shard_dir / f"{stem}_curve_predictions_repeat0.npz",
                shard_dir / f"{stem}_gate_selections.json",
            )
            if not all(path.exists() and path.stat().st_size > 0 for path in required_paths):
                raise ValueError(f"missing downstream artifact for {stem}")
            with np.load(required_paths[2], allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "predictions", "truth", "strain"}:
                    raise ValueError(f"invalid curve archive fields in {required_paths[2]}")
                if not (
                    archive["predictions"].shape
                    == archive["truth"].shape
                    == archive["strain"].shape
                ):
                    raise ValueError(f"curve archive shape mismatch in {required_paths[2]}")
            shards += 1
            split_rows += len(splits)
    return {"downstream_shards": shards, "split_rows": split_rows}


def audit_aggregate(results_dir: Path) -> dict[str, int]:
    aggregate_dir = results_dir / "aggregate"
    required = (
        "aggregation_manifest.json",
        "run_metrics.csv",
        "family_metrics.csv",
        "summary_metrics.csv",
        "summary_family_bootstrap.csv",
        "inference.csv",
        "negative_transfer.csv",
        "scalar_predictions.csv.gz",
        "curve_metrics.csv.gz",
    )
    for name in required:
        path = aggregate_dir / name
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing aggregate artifact: {path}")
    manifest = json.loads((aggregate_dir / "aggregation_manifest.json").read_text(encoding="utf-8"))
    if int(manifest["complete_shards"]) != 40 or int(manifest["expected_shards"]) != 40:
        raise ValueError("aggregate manifest does not report 40/40 shards")
    inference = pd.read_csv(aggregate_dir / "inference.csv")
    frozen = inference[
        inference["comparison_family"].isin(["performance", "mechanism", "attenuation"])
    ]
    if len(frozen) != 3:
        raise ValueError("expected exactly three frozen primary comparisons")
    if frozen[["family_exact_sign_flip_p", "holm_p"]].isna().any().any():
        raise ValueError("missing exact or Holm-adjusted primary inference")
    return {"aggregate_files": len(required), "frozen_comparisons": len(frozen)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = {
        "config_sha256": sha256(args.config),
        **audit_pretraining(args.results_dir, config),
        **audit_downstream(args.results_dir, config),
        **audit_aggregate(args.results_dir),
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
