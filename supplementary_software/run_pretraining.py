"""Train fold-specific SSL encoders under strict geometry-unseen splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from core import (
    PRIMITIVE_NAMES,
    PeriodicEncoder,
    assert_geometry_unseen,
    build_geometry_unseen_cache,
    embed,
    load_config,
    train_variant,
)


def run_fold_seed(
    config_path: Path,
    output_dir: Path,
    held_out_family: int,
    seed: int,
    requested_variants: list[str] | None = None,
) -> None:
    config = load_config(config_path)
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not listed in {config_path}")
    variants = requested_variants or config.variants
    unknown = set(variants) - set(config.variants)
    if unknown:
        raise ValueError(f"variants not enabled by config: {sorted(unknown)}")
    run_root = Path(__file__).resolve().parents[2]
    fold_dir = output_dir / f"fold_{held_out_family}" / f"seed_{seed}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done_path = fold_dir / "pretraining_complete.json"
    completed: set[str] = set()
    if done_path.exists():
        completed = set(json.loads(done_path.read_text(encoding="utf-8"))["completed_variants"])
    payload = build_geometry_unseen_cache(config, run_root, held_out_family, seed)
    assert_geometry_unseen(payload["pretrain_weights"], held_out_family)
    torch.set_num_threads(config.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed + 1009 * held_out_family)
    random_model = PeriodicEncoder(payload["pretrain_descriptors"].shape[1]).to(device)
    random_path = fold_dir / "target_embeddings_random_encoder.npz"
    if not random_path.exists():
        np.savez_compressed(
            random_path,
            target_ids=payload["target_ids"],
            embeddings=embed(random_model, payload["target_volumes"], device),
        )

    history_frames: list[pd.DataFrame] = []
    existing_history = fold_dir / "training_history.csv"
    if existing_history.exists():
        history_frames.append(pd.read_csv(existing_history))
    for variant in variants:
        embedding_path = fold_dir / f"target_embeddings_{variant}.npz"
        model_path = fold_dir / f"model_fold{held_out_family}_seed{seed}_{variant}.pt"
        if variant in completed and embedding_path.exists() and model_path.exists():
            print(json.dumps({"status": "resume_skip", "variant": variant}), flush=True)
            continue
        model, history = train_variant(
            variant,
            config,
            payload["pretrain_volumes"],
            payload["pretrain_descriptors"],
            fold_dir,
            held_out_family,
            seed,
            device,
        )
        np.savez_compressed(
            embedding_path,
            target_ids=payload["target_ids"],
            embeddings=embed(model, payload["target_volumes"], device),
        )
        history_frames.append(history)
        completed.add(variant)
        done_path.write_text(
            json.dumps(
                {
                    "held_out_family": held_out_family,
                    "held_out_name": PRIMITIVE_NAMES[held_out_family],
                    "seed": seed,
                    "device": str(device),
                    "completed_variants": sorted(completed),
                    "strict_geometry_unseen": True,
                    "maximum_held_out_pretraining_weight": float(
                        np.max(np.abs(payload["pretrain_weights"][:, held_out_family]))
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if history_frames:
        history_all = pd.concat(history_frames, ignore_index=True)
        history_all = history_all.drop_duplicates(
            ["variant", "held_out_family", "seed", "epoch"], keep="last"
        ).sort_values(["variant", "epoch"])
        history_all.to_csv(existing_history, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_full") / "pretraining",
    )
    parser.add_argument("--fold", type=int, required=True, choices=range(8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args()
    run_fold_seed(
        args.config.resolve(),
        args.output_dir.resolve(),
        args.fold,
        args.seed,
        args.variants,
    )


if __name__ == "__main__":
    main()
