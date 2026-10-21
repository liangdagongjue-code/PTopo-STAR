"""Generate the frozen Scientific Reports result figures and table fragments."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "explicit_ridge": "Explicit Ridge",
    "explicit_extra_trees": "Explicit ExtraTrees",
    "masked_only_fusion": "Masked-only fusion",
    "periodic_only_fusion": "Periodic-only fusion",
    "topology_only_fusion": "Topology-only fusion",
    "ptopo_fusion": "PTopo fusion",
    "ptopo_residual_ungated": "Ungated residual",
    "ptopo_star": "PTopo-STAR",
    "handcrafted_ridge": "Handcrafted Ridge",
    "weights_plus_handcrafted": "Weights + handcrafted",
    "random_fusion": "Random-encoder fusion",
}

COLORS = {
    "explicit_ridge": "#4C78A8",
    "explicit_extra_trees": "#72B7B2",
    "weights_plus_handcrafted": "#9C755F",
    "masked_only_fusion": "#B279A2",
    "periodic_only_fusion": "#F2CF5B",
    "topology_only_fusion": "#FF9DA6",
    "ptopo_fusion": "#F58518",
    "ptopo_residual_ungated": "#E45756",
    "ptopo_star": "#2E8B57",
}


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
        -0.11,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def figure2_performance(aggregate_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(aggregate_dir / "summary_family_bootstrap.csv")
    tasks = [
        ("energy_dissipation", "Energy dissipation"),
        ("peak_stress", "Peak stress"),
        ("stress_curve", "Stress trace"),
    ]
    methods = [
        "explicit_ridge",
        "explicit_extra_trees",
        "weights_plus_handcrafted",
        "masked_only_fusion",
        "ptopo_fusion",
        "ptopo_star",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharex=True)
    for panel, (ax, (task, title)) in enumerate(zip(axes, tasks, strict=True)):
        frame = summary[summary["task"] == task]
        for method in methods:
            method_frame = frame[frame["method"] == method].sort_values("shots")
            if method_frame.empty:
                continue
            x = method_frame["shots"].to_numpy()
            y = method_frame["macro_nrmse"].to_numpy()
            low = method_frame["ci95_low"].to_numpy()
            high = method_frame["ci95_high"].to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                ms=3.4,
                lw=1.4,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            ax.fill_between(x, low, high, color=COLORS[method], alpha=0.10, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Target-family support samples")
        ax.set_xticks([0, 1, 2, 4])
        ax.grid(axis="y", color="#D9DEE5", lw=0.6)
        panel_label(ax, chr(ord("a") + panel))
    axes[0].set_ylabel("Family-macro NRMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.28, top=0.68)
    save_figure(fig, output_dir / "figure2_performance")


def _heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: Iterable[str],
    column_labels: Iterable[str],
    cmap: str,
    label: str,
    center: float | None = None,
) -> None:
    if center is None:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    else:
        maximum = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))))
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-maximum, vmax=maximum)
    ax.set_yticks(np.arange(len(values)), labels=list(row_labels))
    ax.set_xticks(np.arange(values.shape[1]), labels=list(column_labels))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            color = "white" if abs(value) > 0.65 * np.nanmax(np.abs(values)) else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.047, pad=0.03)
    colorbar.ax.set_title(label, fontsize=6, pad=3)


def figure3_families(aggregate_dir: Path, output_dir: Path) -> None:
    family = pd.read_csv(aggregate_dir / "family_metrics.csv")
    negative = pd.read_csv(aggregate_dir / "negative_transfer.csv")
    energy = family[family["task"] == "energy_dissipation"]
    shots = [1, 2, 4]
    star = (
        energy[(energy["method"] == "ptopo_star") & energy["shots"].isin(shots)]
        .pivot(index="held_out_family", columns="shots", values="nrmse_global_sd")
        .reindex(index=range(8), columns=shots)
    )
    explicit = (
        energy[(energy["method"] == "explicit_ridge") & energy["shots"].isin(shots)]
        .pivot(index="held_out_family", columns="shots", values="nrmse_global_sd")
        .reindex(index=range(8), columns=shots)
    )
    delta = star - explicit
    short_names = [
        "Gyroid",
        "Diamond",
        "Primitive",
        "IWP",
        "Neovius",
        "Fischer-Koch S",
        "FRD",
        "PMY",
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1, 1, 1.05]},
    )
    _heatmap(
        axes[0],
        star.to_numpy(),
        short_names,
        [f"{shot}-shot" for shot in shots],
        "viridis_r",
        "NRMSE",
    )
    axes[0].set_title("Held-out-family error")
    axes[0].set_ylabel("Dominant held-out TPMS family")
    panel_label(axes[0], "a")
    _heatmap(
        axes[1],
        delta.to_numpy(),
        [""] * 8,
        [f"{shot}-shot" for shot in shots],
        "RdBu_r",
        "Delta NRMSE",
        center=0.0,
    )
    axes[1].set_title("Difference vs explicit")
    axes[1].tick_params(axis="y", left=False)
    panel_label(axes[1], "b")
    methods = ["ptopo_fusion", "ptopo_residual_ungated", "ptopo_star"]
    frame = negative[
        (negative["task"] == "energy_dissipation")
        & negative["shots"].isin(shots)
    ]
    for method in methods:
        item = frame[frame["method"] == method].sort_values("shots")
        axes[2].plot(
            item["shots"],
            item["negative_transfer_rate"],
            marker="o",
            ms=4,
            lw=1.5,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[2].axhline(0.5, color="#A0A6AD", ls="--", lw=0.8)
    axes[2].set_xticks(shots)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Target-family support samples")
    axes[2].set_ylabel("Negative-transfer rate")
    axes[2].set_title("Negative transfer")
    axes[2].grid(axis="y", color="#D9DEE5", lw=0.6)
    axes[2].legend(frameon=False, loc="upper right")
    panel_label(axes[2], "c")
    fig.subplots_adjust(wspace=0.48)
    save_figure(fig, output_dir / "figure3_families")


def load_curve_archive(shard_dir: Path) -> tuple[np.ndarray, ...]:
    metadata: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    strain: list[np.ndarray] = []
    paths = sorted(shard_dir.glob("*_curve_predictions_repeat0.npz"))
    if not paths:
        raise ValueError(f"no curve prediction archives under {shard_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"metadata", "predictions", "truth", "strain"}
            if not required.issubset(data.files):
                raise ValueError(f"{path} predates the self-contained curve archive format")
            metadata.append(data["metadata"])
            prediction.append(data["predictions"])
            truth.append(data["truth"])
            strain.append(data["strain"])
    return (
        np.concatenate(metadata),
        np.concatenate(prediction),
        np.concatenate(truth),
        np.concatenate(strain),
    )


def figure4_gate_and_curves(
    aggregate_dir: Path,
    shard_dir: Path,
    output_dir: Path,
) -> None:
    scalar = pd.read_csv(aggregate_dir / "scalar_predictions.csv.gz")
    available_shots = sorted(
        scalar.loc[scalar["task"] == "energy_dissipation", "shots"].unique().tolist()
    )
    curve_shot = 2 if 2 in available_shots else int(available_shots[0])
    star = scalar[
        (scalar["task"] == "energy_dissipation")
        & (scalar["shots"] == curve_shot)
        & (scalar["method"] == "ptopo_star")
    ].copy()
    explicit = scalar[
        (scalar["task"] == "energy_dissipation")
        & (scalar["shots"] == curve_shot)
        & (scalar["method"] == "explicit_ridge")
    ].copy()
    keys = ["held_out_family", "sample_id", "seed", "shots", "repeat"]
    paired = star.merge(
        explicit[[*keys, "absolute_error"]],
        on=keys,
        suffixes=("_star", "_explicit"),
    )
    paired["delta_absolute_error_sd"] = (
        paired["absolute_error_star"] - paired["absolute_error_explicit"]
    ) / paired["normalizer_sd"]
    sample = (
        paired.groupby(["held_out_family", "sample_id"], as_index=False)
        .agg(
            gate=("gate", "mean"),
            distance=("distance_norm", "mean"),
            uncertainty=("uncertainty_norm", "mean"),
            delta_error=("delta_absolute_error_sd", "mean"),
        )
    )
    metadata, predictions, truth, strain = load_curve_archive(shard_dir)
    curve_seed = int(np.median(np.unique(metadata["seed"])))
    curve_families = [0, 4, 5, 7]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    scatter = axes[0, 0].scatter(
        sample["distance"],
        sample["gate"],
        c=sample["uncertainty"],
        s=15,
        cmap="plasma",
        alpha=0.75,
        edgecolors="none",
    )
    axes[0, 0].axvline(1.0, color="#8C8C8C", ls="--", lw=0.8)
    axes[0, 0].set_xlabel("Normalized representation distance")
    axes[0, 0].set_ylabel("Mean residual gate")
    axes[0, 0].set_title("Residual gate")
    colorbar_axis = axes[0, 0].inset_axes([0.57, 0.08, 0.34, 0.035])
    colorbar = fig.colorbar(scatter, cax=colorbar_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5, length=1.5, pad=1)
    colorbar.ax.set_title("Disagreement", fontsize=5, pad=1)
    panel_label(axes[0, 0], "a")
    axes[0, 1].scatter(
        sample["gate"],
        sample["delta_error"],
        color=COLORS["ptopo_star"],
        s=15,
        alpha=0.75,
        edgecolors="none",
    )
    axes[0, 1].axhline(0.0, color="#8C8C8C", ls="--", lw=0.8)
    axes[0, 1].set_xlabel("Mean residual gate")
    axes[0, 1].set_ylabel("Delta absolute error / global SD")
    axes[0, 1].set_title("Gate and error change")
    panel_label(axes[0, 1], "b")

    curve_axes = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for panel_index, (ax, family) in enumerate(
        zip(curve_axes, curve_families, strict=True), start=2
    ):
        mask = (
            (metadata["seed"] == curve_seed)
            & (metadata["held_out_family"] == family)
            & (metadata["shots"] == curve_shot)
        )
        family_metadata = metadata[mask]
        candidate_ids = sorted(
            set(family_metadata[family_metadata["method"] == "explicit_ridge"]["sample_id"])
            & set(family_metadata[family_metadata["method"] == "ptopo_star"]["sample_id"])
        )
        if not candidate_ids:
            raise ValueError(f"no deterministic curve candidate for family {family}")
        sample_id = candidate_ids[0]
        local_indices = np.flatnonzero(mask & (metadata["sample_id"] == sample_id))
        index_by_method = {metadata[index]["method"]: index for index in local_indices}
        true_index = index_by_method["ptopo_star"]
        ax.plot(
            strain[true_index],
            truth[true_index],
            color="#222222",
            lw=1.7,
            label="Measured",
        )
        for method in ("explicit_ridge", "ptopo_star"):
            index = index_by_method[method]
            ax.plot(
                strain[index],
                predictions[index],
                color=COLORS[method],
                lw=1.25,
                ls="--" if method == "explicit_ridge" else "-",
                label=METHOD_LABELS[method],
            )
        short_name = {
            0: "Gyroid",
            4: "Neovius",
            5: "Fischer-Koch S",
            7: "PMY",
        }[family]
        ax.set_title(f"{short_name} target")
        ax.set_xlabel("Engineering strain")
        ax.set_ylabel("Stress (MPa)")
        ax.grid(color="#E3E6EA", lw=0.5)
        panel_label(ax, chr(ord("a") + panel_index))
    handles, labels = curve_axes[-1].get_legend_handles_labels()
    curve_axes[-1].legend(handles, labels, frameon=False, loc="best")
    fig.subplots_adjust(wspace=0.42, hspace=0.48)
    save_figure(fig, output_dir / "figure4_gate_curves")


def _format_ci(row: pd.Series) -> str:
    return f"{row['macro_nrmse']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"


def write_tables(aggregate_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(aggregate_dir / "summary_family_bootstrap.csv")
    family = pd.read_csv(aggregate_dir / "family_metrics.csv")
    inference = pd.read_csv(aggregate_dir / "inference.csv")
    negative = pd.read_csv(aggregate_dir / "negative_transfer.csv")

    counts = (
        family[
            (family["task"] == "energy_dissipation")
            & (family["method"] == "explicit_ridge")
            & (family["shots"] == family["shots"].min())
        ][["held_out_family", "held_out_name"]]
        .drop_duplicates()
        .sort_values("held_out_family")
    )
    count_map = {0: 12, 1: 15, 2: 9, 3: 22, 4: 8, 5: 124, 6: 44, 7: 40}
    counts["physical_samples"] = counts["held_out_family"].map(count_map)
    counts.to_csv(output_dir / "table1_family_counts.csv", index=False)

    methods = [
        "explicit_ridge",
        "explicit_extra_trees",
        "weights_plus_handcrafted",
        "masked_only_fusion",
        "ptopo_fusion",
        "ptopo_star",
    ]
    energy = summary[
        (summary["task"] == "energy_dissipation")
        & summary["method"].isin(methods)
        & summary["shots"].isin([1, 2, 4])
    ].copy()
    energy["value"] = energy.apply(_format_ci, axis=1)
    table2 = energy.pivot(index="method", columns="shots", values="value").reindex(methods)
    table2.insert(0, "method_label", [METHOD_LABELS[item] for item in table2.index])
    table2.to_csv(output_dir / "table2_energy_main.csv")

    ablations = [
        "masked_only_fusion",
        "periodic_only_fusion",
        "topology_only_fusion",
        "ptopo_fusion",
        "ptopo_residual_ungated",
        "ptopo_star",
    ]
    family_pool = family[
        (family["task"] == "energy_dissipation")
        & family["shots"].isin([1, 2, 4])
        & family["method"].isin(ablations)
    ]
    ablation_summary = (
        family_pool.groupby("method", as_index=False)
        .agg(pooled_family_macro_nrmse=("nrmse_global_sd", "mean"))
        .set_index("method")
        .reindex(ablations)
        .reset_index()
    )
    negative_pool = negative[
        (negative["task"] == "energy_dissipation")
        & negative["shots"].isin([1, 2, 4])
        & negative["method"].isin(ablations)
    ]
    negative_pool = negative_pool.groupby("method", as_index=False).agg(
        negative_transfer_rate=("negative_transfer_rate", "mean")
    )
    table3 = ablation_summary.merge(negative_pool, on="method", how="left")
    table3.insert(1, "method_label", table3["method"].map(METHOD_LABELS))
    table3.to_csv(output_dir / "table3_ablation_gate.csv", index=False)
    inference.to_csv(output_dir / "table_primary_inference.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_style()
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figure2_performance(args.aggregate_dir, figures)
    figure3_families(args.aggregate_dir, figures)
    figure4_gate_and_curves(args.aggregate_dir, args.shard_dir, figures)
    write_tables(args.aggregate_dir, tables)


if __name__ == "__main__":
    main()
