"""Resumable parallel orchestration for all 8 folds and 5 seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core import load_config


def run_one(command: list[str], log_path: Path) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return {
        "command": command,
        "log": str(log_path),
        "returncode": result.returncode,
    }


def run_stage(
    stage: str,
    config_path: Path,
    results_dir: Path,
    workers: int,
) -> list[dict[str, object]]:
    config = load_config(config_path)
    commands: list[tuple[list[str], Path]] = []
    logs = results_dir / "orchestration_logs"
    for fold in range(8):
        for seed in config.seeds:
            if stage == "pretraining":
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("run_pretraining.py")),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(results_dir / "pretraining"),
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                ]
            elif stage == "downstream":
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("run_downstream.py")),
                    "--config",
                    str(config_path),
                    "--pretraining-dir",
                    str(results_dir / "pretraining"),
                    "--output-dir",
                    str(results_dir / "downstream_shards"),
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                ]
            else:
                raise ValueError(stage)
            commands.append((command, logs / f"{stage}_fold{fold}_seed{seed}.log"))
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, command, log_path): (command, log_path)
            for command, log_path in commands
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    failures = [row for row in rows if row["returncode"] != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} {stage} jobs failed; inspect orchestration logs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pretraining", "downstream", "all"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).with_name("results_full"),
    )
    parser.add_argument("--pretraining-workers", type=int, default=6)
    parser.add_argument("--downstream-workers", type=int, default=1)
    args = parser.parse_args()
    config_path = args.config.resolve()
    results_dir = args.results_dir.resolve()
    if args.stage in {"pretraining", "all"}:
        run_stage("pretraining", config_path, results_dir, args.pretraining_workers)
    if args.stage in {"downstream", "all"}:
        run_stage("downstream", config_path, results_dir, args.downstream_workers)


if __name__ == "__main__":
    main()
