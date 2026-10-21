"""Generate LaTeX table fragments directly from frozen aggregate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHOD_LABELS = {
    "explicit_ridge": "Explicit Ridge",
    "explicit_extra_trees": "Explicit ExtraTrees",
    "handcrafted_ridge": "Handcrafted Ridge",
    "weights_plus_handcrafted": r"Weights $+$ handcrafted",
    "random_fusion": "Random-encoder fusion",
    "masked_only_fusion": "Masked-only fusion",
    "periodic_only_fusion": "Periodic-only fusion",
    "topology_only_fusion": "Topology-only fusion",
    "ptopo_fusion": "PTopo fusion",
    "ptopo_residual_ungated": "Ungated PTopo residual",
    "ptopo_star": "PTopo-STAR",
}

TASK_LABELS = {
    "energy_dissipation": "Energy dissipation",
    "peak_stress": "Peak stress",
    "stress_curve": "Stress trace",
}

ABLATION_LABELS = {
    "masked_only_fusion": "mask reconstruction",
    "periodic_only_fusion": r"mask $+$ periodic",
    "topology_only_fusion": r"mask $+$ topology",
    "ptopo_fusion": r"mask $+$ periodic $+$ topology",
    "ptopo_residual_ungated": "PTopo nested residual, no attenuation",
    "ptopo_star": "PTopo nested residual, source-only gate",
}


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ci_value(row: pd.Series) -> str:
    return f"{row['macro_nrmse']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"


def p_value(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def table2_rows(summary: pd.DataFrame, output_dir: Path) -> None:
    methods = [
        "explicit_ridge",
        "explicit_extra_trees",
        "weights_plus_handcrafted",
        "masked_only_fusion",
        "ptopo_fusion",
        "ptopo_star",
    ]
    frame = summary[
        (summary["task"] == "energy_dissipation")
        & summary["method"].isin(methods)
        & summary["shots"].isin([1, 2, 4])
    ].copy()
    frame["value"] = frame.apply(ci_value, axis=1)
    pivot = frame.pivot(index="method", columns="shots", values="value")
    missing = [(method, shot) for method in methods for shot in (1, 2, 4) if pd.isna(pivot.loc[method, shot])]
    if missing:
        raise ValueError(f"missing Table 2 cells: {missing}")
    lines = [
        f"{METHOD_LABELS[method]} & {pivot.loc[method, 1]} & {pivot.loc[method, 2]} & {pivot.loc[method, 4]} \\\\"
        for method in methods
    ]
    write_lines(output_dir / "table2_energy_rows.tex", lines)


def table3_rows(
    family: pd.DataFrame,
    negative: pd.DataFrame,
    output_dir: Path,
) -> None:
    methods = list(ABLATION_LABELS)
    pooled = (
        family[
            (family["task"] == "energy_dissipation")
            & family["shots"].isin([1, 2, 4])
            & family["method"].isin(methods)
        ]
        .groupby("method", as_index=False)
        .agg(macro_nrmse=("nrmse_global_sd", "mean"))
        .set_index("method")
    )
    neg = (
        negative[
            (negative["task"] == "energy_dissipation")
            & negative["shots"].isin([1, 2, 4])
            & negative["method"].isin(methods)
        ]
        .groupby("method", as_index=False)
        .agg(rate=("negative_transfer_rate", "mean"))
        .set_index("method")
    )
    lines = []
    for method in methods:
        lines.append(
            f"{METHOD_LABELS[method]} & {ABLATION_LABELS[method]} & "
            f"{pooled.loc[method, 'macro_nrmse']:.3f} & {100 * neg.loc[method, 'rate']:.1f}\\% \\\\"
        )
    write_lines(output_dir / "table3_ablation_rows.tex", lines)


def supplement_primary_inference(inference: pd.DataFrame, output_dir: Path) -> None:
    frozen = inference[inference["comparison_family"].isin(["performance", "mechanism", "attenuation"])].copy()
    if len(frozen) != 3:
        raise ValueError(f"expected three frozen inference rows, found {len(frozen)}")
    lines = [
        r"\begin{longtable}{p{29mm}p{29mm}rrrr}",
        r"\caption{Frozen primary energy-dissipation comparisons pooled over 1/2/4-shot strata. Effects are method minus reference family-macro NRMSE; negative values favour the named method.}\label{tab:s4_inference}\\",
        r"\toprule",
        r"Method & Reference & Effect & 95\% CI & Exact $p$ & Holm $p$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Method & Reference & Effect & 95\% CI & Exact $p$ & Holm $p$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in frozen.iterrows():
        lines.append(
            f"{METHOD_LABELS[row['method']]} & {METHOD_LABELS[row['reference']]} & "
            f"{row['effect_nrmse_method_minus_reference']:.3f} & "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] & "
            f"{p_value(row['family_exact_sign_flip_p'])} & {p_value(row['holm_p'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write_lines(output_dir / "supp_table_s4_primary_inference.tex", lines)


def supplement_all_results(summary: pd.DataFrame, output_dir: Path) -> None:
    methods = list(METHOD_LABELS)
    tasks = list(TASK_LABELS)
    frame = summary[
        summary["task"].isin(tasks)
        & summary["method"].isin(methods)
        & summary["shots"].isin([0, 1, 2, 4])
    ].copy()
    frame["value"] = frame.apply(ci_value, axis=1)
    pivot = frame.pivot(index=["task", "method"], columns="shots", values="value")
    lines = [
        r"\begin{longtable}{p{20mm}p{30mm}p{22mm}p{22mm}p{22mm}p{22mm}}",
        r"\caption{Family-macro NRMSE for every downstream method, outcome, and support size. Brackets give descriptive 95\% held-out-family bootstrap intervals after algorithmic repetitions were averaged.}\label{tab:s5_all_results}\\",
        r"\toprule",
        r"Outcome & Method & 0-shot & 1-shot & 2-shot & 4-shot \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Outcome & Method & 0-shot & 1-shot & 2-shot & 4-shot \\",
        r"\midrule",
        r"\endhead",
    ]
    for task in tasks:
        for method in methods:
            key = (task, method)
            if key not in pivot.index:
                raise ValueError(f"missing Supplementary Table S5 row: {key}")
            row = pivot.loc[key]
            values = [row[shot] for shot in (0, 1, 2, 4)]
            if any(pd.isna(value) for value in values):
                raise ValueError(f"missing Supplementary Table S5 cells: {key}")
            lines.append(
                f"{TASK_LABELS[task]} & {METHOD_LABELS[method]} & "
                + " & ".join(values)
                + r" \\"
            )
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write_lines(output_dir / "supp_table_s5_all_results.tex", lines)


def supplement_gate_table(gate_frequency_path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(gate_frequency_path).sort_values(
        ["task", "selections", "scale", "lambda_distance", "lambda_uncertainty"],
        ascending=[True, False, True, True, True],
    )
    lines = [
        r"\begin{longtable}{p{27mm}rrrrrr}",
        r"\caption{Frequency and source-validation diagnostics for all selected gate settings across 40 fold--seed runs per outcome.}\label{tab:s3_gate}\\",
        r"\toprule",
        r"Outcome & $s$ & $\lambda_d$ & $\lambda_u$ & Count & Source NT & Mean gate \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Outcome & $s$ & $\lambda_d$ & $\lambda_u$ & Count & Source NT & Mean gate \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"{TASK_LABELS[row['task']]} & {int(row['scale'])} & "
            f"{row['lambda_distance']:.2f} & {row['lambda_uncertainty']:.2f} & "
            f"{int(row['selections'])} & {100 * row['mean_source_negative_transfer']:.1f}\\% & "
            f"{row['mean_selected_gate']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    write_lines(output_dir / "supp_table_s3_gate.tex", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--gate-frequency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.aggregate_dir / "summary_family_bootstrap.csv")
    family = pd.read_csv(args.aggregate_dir / "family_metrics.csv")
    inference = pd.read_csv(args.aggregate_dir / "inference.csv")
    negative = pd.read_csv(args.aggregate_dir / "negative_transfer.csv")

    table2_rows(summary, args.output_dir)
    table3_rows(family, negative, args.output_dir)
    supplement_primary_inference(inference, args.output_dir)
    supplement_all_results(summary, args.output_dir)
    supplement_gate_table(args.gate_frequency, args.output_dir)


if __name__ == "__main__":
    main()
