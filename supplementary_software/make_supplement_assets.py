"""Generate supplementary figures and source tables from a completed formal run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import PRIMITIVE_NAMES

VARIANT_LABELS = {
    "masked_only": "Masked-only",
    "periodic_only": "Periodic-only",
    "topology_only": "Topology-only",
    "ptopo_ssl": "PTopo",
}

VARIANT_COLORS = {
    "masked_only": "#B279A2",
    "periodic_only": "#F2CF5B",
    "topology_only": "#FF9DA6",
    "ptopo_ssl": "#F58518",
}

SHORT_FAMILIES = [
    "Gyroid",
    "Diamond",
    "Primitive",
    "IWP",
    "Neovius",
    "Fischer-Koch S",
    "FRD",
    "PMY",
]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def read_training_histories(pretraining_dir: Path) -> pd.DataFrame:
    paths = sorted(pretraining_dir.glob("fold_*/seed_*/training_history.csv"))
    if len(paths) != 40:
        raise ValueError(f"expected 40 training histories, found {len(paths)}")
    history = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected_variants = set(VARIANT_LABELS)
    if set(history["variant"].unique()) != expected_variants:
        raise ValueError("training histories do not contain the four frozen variants")
    return history


def _plot_loss(
    ax: plt.Axes,
    history: pd.DataFrame,
    column: str,
    variants: list[str],
    title: str,
) -> None:
    frame = history[history["variant"].isin(variants)]
    grouped = frame.groupby(["variant", "epoch"])[column]
    mean = grouped.mean().rename("mean").reset_index()
    quantiles = grouped.quantile([0.025, 0.975]).unstack().reset_index()
    quantiles.columns = ["variant", "epoch", "low", "high"]
    summary = mean.merge(quantiles, on=["variant", "epoch"])
    for variant in variants:
        item = summary[summary["variant"] == variant].sort_values("epoch")
        ax.plot(
            item["epoch"],
            item["mean"],
            color=VARIANT_COLORS[variant],
            lw=1.35,
            label=VARIANT_LABELS[variant],
        )
        ax.fill_between(
            item["epoch"],
            item["low"],
            item["high"],
            color=VARIANT_COLORS[variant],
            alpha=0.12,
            lw=0,
        )
    ax.set_title(title)
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Loss")
    ax.grid(axis="y", color="#D9DEE5", lw=0.6)


def figure_s1_training(history: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    _plot_loss(
        axes[0],
        history,
        "reconstruction_loss",
        list(VARIANT_LABELS),
        "Masked reconstruction",
    )
    _plot_loss(
        axes[1],
        history,
        "periodic_consistency_loss",
        ["periodic_only", "ptopo_ssl"],
        "Periodic-shift consistency",
    )
    _plot_loss(
        axes[2],
        history,
        "topology_loss",
        ["topology_only", "ptopo_ssl"],
        "Topology descriptors",
    )
    for index, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + index))
        ax.legend(frameon=False)
    fig.subplots_adjust(wspace=0.34)
    save_figure(fig, output_dir / "figure_s1_training_history")


def figure_s2_family_effects(aggregate_dir: Path, output_dir: Path) -> pd.DataFrame:
    family = pd.read_csv(aggregate_dir / "family_metrics.csv")
    family = family[family["shots"].isin([1, 2, 4])]
    pooled = family.groupby(
        ["task", "method", "held_out_family", "held_out_name"], as_index=False
    ).agg(nrmse=("nrmse_global_sd", "mean"))
    star = pooled[pooled["method"] == "ptopo_star"]
    explicit = pooled[pooled["method"] == "explicit_ridge"]
    effects = star.merge(
        explicit,
        on=["task", "held_out_family", "held_out_name"],
        suffixes=("_star", "_explicit"),
    )
    effects["delta_nrmse"] = effects["nrmse_star"] - effects["nrmse_explicit"]
    tasks = [
        ("energy_dissipation", "Energy dissipation"),
        ("peak_stress", "Peak stress"),
        ("stress_curve", "Stress trace"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    for index, (ax, (task, title)) in enumerate(zip(axes, tasks, strict=True)):
        item = effects[effects["task"] == task].set_index("held_out_family").reindex(range(8))
        values = item["delta_nrmse"].to_numpy()
        colors = np.where(values <= 0, "#2E8B57", "#E45756")
        ax.barh(np.arange(8), values, color=colors, height=0.67)
        ax.axvline(0.0, color="#666666", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("PTopo-STAR minus explicit NRMSE")
        ax.grid(axis="x", color="#D9DEE5", lw=0.6)
        panel_label(ax, chr(ord("a") + index))
    axes[0].set_yticks(np.arange(8), labels=SHORT_FAMILIES)
    axes[0].invert_yaxis()
    fig.subplots_adjust(wspace=0.28)
    save_figure(fig, output_dir / "figure_s2_family_effects")
    return effects


def summarize_gate_selections(shard_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    paths = sorted(shard_dir.glob("fold*_seed*_gate_selections.json"))
    if len(paths) != 40:
        raise ValueError(f"expected 40 gate-selection files, found {len(paths)}")
    for path in paths:
        stem = path.stem.replace("_gate_selections", "")
        fold_text, seed_text = stem.split("_")
        fold = int(fold_text.removeprefix("fold"))
        seed = int(seed_text.removeprefix("seed"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task, selection in payload.items():
            rows.append(
                {
                    "fold": fold,
                    "held_out_name": PRIMITIVE_NAMES[fold],
                    "seed": seed,
                    "task": task,
                    **selection,
                }
            )
    selections = pd.DataFrame(rows)
    frequencies = (
        selections.groupby(
            ["task", "scale", "lambda_distance", "lambda_uncertainty"],
            as_index=False,
        )
        .agg(
            selections=("seed", "size"),
            mean_source_negative_transfer=("source_negative_transfer_rate", "mean"),
            mean_selected_gate=("selected_mean_gate", "mean"),
        )
        .sort_values(["task", "selections"], ascending=[True, False])
    )
    return selections, frequencies


def write_source_tables(
    aggregate_dir: Path,
    history: pd.DataFrame,
    family_effects: pd.DataFrame,
    selections: pd.DataFrame,
    frequencies: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "training_history.csv.gz", index=False, compression="gzip")
    family_effects.to_csv(output_dir / "family_effects_pooled_1_2_4_shot.csv", index=False)
    selections.to_csv(output_dir / "gate_selections.csv", index=False)
    frequencies.to_csv(output_dir / "gate_selection_frequencies.csv", index=False)
    for name in (
        "run_metrics.csv",
        "summary_metrics.csv",
        "summary_family_bootstrap.csv",
        "family_metrics.csv",
        "inference.csv",
        "negative_transfer.csv",
        "aggregation_manifest.json",
    ):
        source = aggregate_dir / name
        target = output_dir / name
        target.write_bytes(source.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretraining-dir", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    figures = args.output_dir / "figures"
    tables = args.output_dir / "derived_data"
    history = read_training_histories(args.pretraining_dir)
    figure_s1_training(history, figures)
    family_effects = figure_s2_family_effects(args.aggregate_dir, figures)
    selections, frequencies = summarize_gate_selections(args.shard_dir)
    write_source_tables(
        args.aggregate_dir,
        history,
        family_effects,
        selections,
        frequencies,
        tables,
    )


if __name__ == "__main__":
    main()
