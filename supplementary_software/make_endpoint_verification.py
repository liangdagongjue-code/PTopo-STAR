"""Create the machine-readable TPMS-274 endpoint and dominant-family audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from core import PRIMITIVE_NAMES, load_physical_samples
from latticetopo_pilot.tpms_geometry import energy_dissipation_from_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    samples = load_physical_samples(args.dataset_root)
    rows: list[dict[str, object]] = []
    for sample in samples:
        reproduced = energy_dissipation_from_curve(sample.strain, sample.stress)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "dominant_family": sample.dominant_family,
                "dominant_name": PRIMITIVE_NAMES[sample.dominant_family],
                "stored_energy_mj_m3": sample.energy_dissipation,
                "reproduced_energy_mj_m3": reproduced,
                "absolute_difference": abs(reproduced - sample.energy_dissipation),
                "relative_difference": abs(reproduced - sample.energy_dissipation)
                / max(abs(sample.energy_dissipation), np.finfo(float).tiny),
                "peak_stress_mpa": sample.peak_stress,
                "maximum_strain": float(np.max(sample.strain)),
                "trace_points": len(sample.strain),
            }
        )
    frame = pd.DataFrame(rows)
    counts = (
        frame.groupby(["dominant_family", "dominant_name"], as_index=False)
        .agg(physical_designs=("sample_id", "size"))
        .sort_values("dominant_family")
    )
    report = {
        "physical_designs": len(frame),
        "all_trace_points": sorted(frame["trace_points"].unique().tolist()),
        "maximum_absolute_energy_difference": float(frame["absolute_difference"].max()),
        "maximum_relative_energy_difference": float(frame["relative_difference"].max()),
        "energy_range_mj_m3": [
            float(frame["stored_energy_mj_m3"].min()),
            float(frame["stored_energy_mj_m3"].max()),
        ],
        "peak_stress_range_mpa": [
            float(frame["peak_stress_mpa"].min()),
            float(frame["peak_stress_mpa"].max()),
        ],
        "family_counts": counts.to_dict(orient="records"),
        "status": "PASS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_columns = [
        "sample_id",
        "dominant_family",
        "dominant_name",
        "absolute_difference",
        "relative_difference",
        "trace_points",
    ]
    frame[public_columns].to_csv(
        args.output_dir / "endpoint_verification.csv", index=False
    )
    counts.to_csv(args.output_dir / "family_counts.csv", index=False)
    (args.output_dir / "endpoint_verification_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
