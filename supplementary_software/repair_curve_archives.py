"""Add ground-truth stress and strain arrays to legacy curve archives.

This utility is intended for controlled recovery when prediction archives were
written before the self-contained Figure 4 format was introduced.  It does not
alter metadata or model predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curves(dataset_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(dataset_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if "strain" not in payload or "stress" not in payload:
            continue
        curves[path.stem] = (
            np.asarray(payload["strain"], dtype=np.float32),
            np.asarray(payload["stress"], dtype=np.float32),
        )
    if len(curves) != 274:
        raise ValueError(f"expected 274 physical curves, found {len(curves)}")
    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    curves = load_curves(args.dataset_root)
    records: list[dict[str, object]] = []
    for path in sorted(args.shard_dir.glob("*_curve_predictions_repeat0.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = archive["metadata"]
            predictions = archive["predictions"]
            existing = set(archive.files)
            if existing == {"metadata", "predictions", "truth", "strain"}:
                records.append({"file": path.name, "status": "already_complete", "sha256": sha256(path)})
                continue
            if existing != {"metadata", "predictions"}:
                raise ValueError(f"unexpected fields in {path}: {sorted(existing)}")
        sample_ids = metadata["sample_id"].tolist()
        strain = np.stack([curves[str(sample_id)][0] for sample_id in sample_ids])
        truth = np.stack([curves[str(sample_id)][1] for sample_id in sample_ids])
        if not (predictions.shape == truth.shape == strain.shape):
            raise ValueError(f"shape mismatch in {path}")
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            metadata=metadata,
            predictions=predictions,
            truth=truth,
            strain=strain,
        )
        temporary.replace(path)
        records.append({"file": path.name, "status": "repaired", "sha256": sha256(path)})

    if len(records) != 8:
        raise ValueError(f"expected eight curve archives, found {len(records)}")
    report = {"status": "PASS", "archives": len(records), "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "archives": len(records)}, indent=2))


if __name__ == "__main__":
    main()
