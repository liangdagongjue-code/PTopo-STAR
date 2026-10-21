"""Reconstruct logical formal scalar-prediction shards from the frozen aggregate.

The original per-fold scalar CSV files were not copied back from the remote run.
The frozen aggregate preserves every scalar row together with its fold and seed,
so the logical shards can be recovered without fitting a model or changing any
reported value.  The compressed byte streams are newly generated and therefore
are not claimed to be byte-identical to the original remote files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_deterministic_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.aggregate)
    required = {"held_out_family", "seed"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"aggregate is missing required columns: {missing}")

    records: list[dict[str, object]] = []
    recovered_rows = 0
    groups = frame.groupby(["held_out_family", "seed"], sort=True, dropna=False)
    for (family, seed), shard in groups:
        family_int = int(family)
        seed_int = int(seed)
        output = args.output_dir / f"fold{family_int}_seed{seed_int}_scalar_predictions.csv.gz"
        shard = shard.reset_index(drop=True)
        write_deterministic_gzip(shard, output)
        recovered_rows += len(shard)
        records.append(
            {
                "file": output.name,
                "held_out_family": family_int,
                "seed": seed_int,
                "rows": len(shard),
                "sha256": sha256(output),
            }
        )

    if len(records) != 40:
        raise ValueError(f"expected 40 fold/seed shards, recovered {len(records)}")
    if recovered_rows != len(frame):
        raise ValueError(f"row mismatch: source={len(frame)}, recovered={recovered_rows}")

    report = {
        "status": "PASS",
        "provenance": "logical reconstruction from frozen formal aggregate",
        "byte_identity_claimed": False,
        "source_file": args.aggregate.name,
        "source_sha256": sha256(args.aggregate),
        "source_rows": len(frame),
        "recovered_shards": len(records),
        "recovered_rows": recovered_rows,
        "shards": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "shards"}, indent=2))


if __name__ == "__main__":
    main()
