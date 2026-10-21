from __future__ import annotations

import numpy as np
import pandas as pd
from aggregate_results import (
    exact_sign_flip_pvalue,
    family_bootstrap_summary,
    family_effects,
    hierarchical_paired_bootstrap,
    holm_adjust,
)


def collapsed_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "held_out_family": np.repeat(np.arange(4), 5),
            "sample_id": [f"s{i}" for i in range(20)],
            "method_mse": np.full(20, 0.25),
            "reference_mse": np.full(20, 1.0),
            "normalizer": np.full(20, 2.0),
        }
    )


def test_family_effects_use_rmse_difference() -> None:
    effects = family_effects(collapsed_table())
    np.testing.assert_allclose(effects, -0.25)


def test_hierarchical_bootstrap_is_deterministic() -> None:
    first = hierarchical_paired_bootstrap(collapsed_table(), 100, seed=7)
    second = hierarchical_paired_bootstrap(collapsed_table(), 100, seed=7)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, (-0.25, -0.25, -0.25))


def test_exact_sign_flip_and_holm() -> None:
    pvalue = exact_sign_flip_pvalue(np.asarray([-1.0, -1.0, -1.0, -1.0]))
    assert pvalue == 0.125
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_family_bootstrap_summary_preserves_macro_point_estimate() -> None:
    frame = pd.DataFrame(
        {
            "task": ["energy"] * 3,
            "method": ["model"] * 3,
            "shots": [1] * 3,
            "nrmse_global_sd": [1.0, 2.0, 3.0],
        }
    )
    result = family_bootstrap_summary(frame, replicates=200, seed=7).iloc[0]
    assert result["families"] == 3
    assert result["macro_nrmse"] == 2.0
    assert result["ci95_low"] <= 2.0 <= result["ci95_high"]
