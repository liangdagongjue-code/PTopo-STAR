from __future__ import annotations

import numpy as np
from run_downstream import curve_energy, curve_metrics, support_indices


def test_support_indices_are_method_independent_and_reproducible() -> None:
    target = np.arange(20)
    first = support_indices(target, 4, 3, 2, 17)
    second = support_indices(target, 4, 3, 2, 17)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 4


def test_curve_energy_matches_known_loop() -> None:
    strain = np.asarray([[0.0, 0.5, 1.0, 1.0, 0.5, 0.0]])
    curve = np.asarray([[0.0, 1.0, 2.0, 1.0, 0.5, 0.0]])
    expected = np.trapezoid(curve[:, :3] - curve[:, 3:][:, ::-1], dx=1.0 / 3, axis=1)
    np.testing.assert_allclose(curve_energy(curve, strain), expected)


def test_curve_metrics_are_zero_for_exact_prediction() -> None:
    curve = np.arange(12, dtype=float).reshape(2, 6)
    metrics = curve_metrics(curve, curve, global_stress_sd=2.0)
    assert all(np.allclose(values, 0.0) for values in metrics.values())
