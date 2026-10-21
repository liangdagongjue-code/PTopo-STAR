from __future__ import annotations

import numpy as np
from downstream import (
    crossfit_base_residuals,
    fit_predict_ridge,
    fit_residual_system,
    gate_values,
    select_gate_source_only,
)


def synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(4), 8)
    explicit = rng.normal(size=(len(groups), 3))
    latent = np.column_stack([explicit[:, 0] ** 2, groups, rng.normal(size=len(groups))])
    y = 0.7 * explicit[:, 0] - 0.2 * explicit[:, 1] + 0.3 * latent[:, 0]
    return explicit, latent, y, groups


def test_crossfit_residuals_have_expected_shape() -> None:
    explicit, _, y, groups = synthetic_problem()
    residual = crossfit_base_residuals(explicit, y, groups, [0.01, 1.0])
    assert residual.shape == y.shape
    assert np.isfinite(residual).all()


def test_gate_is_bounded_and_monotone() -> None:
    distance = np.asarray([0.0, 1.0, 1.5, 2.0])
    uncertainty = np.zeros(4)
    gate = gate_values(distance, uncertainty, 1.0, 0.5, 0.5)
    assert np.all((gate >= 0.0) & (gate <= 1.0))
    np.testing.assert_allclose(gate[:2], 1.0)
    assert np.all(np.diff(gate[1:]) < 0.0)


def test_residual_system_and_source_only_selection() -> None:
    explicit, latent, y, groups = synthetic_problem()
    system = fit_residual_system(
        explicit[:-4],
        latent[:-4],
        y[:-4],
        groups[:-4],
        explicit[-4:],
        latent[-4:],
        [0.01, 1.0],
        ensemble_size=3,
        seed=9,
    )
    assert system.base.shape == (4,)
    assert system.residual_mean.shape == (4,)
    assert np.isfinite(system.distance_norm).all()
    fused = fit_predict_ridge(
        np.concatenate([explicit[:-4], latent[:-4]], axis=1),
        y[:-4],
        np.concatenate([explicit[-4:], latent[-4:]], axis=1),
        [0.01, 1.0],
    )
    np.testing.assert_allclose(system.base + system.residual_mean, fused)
    selection, rows = select_gate_source_only(
        explicit,
        latent,
        y,
        groups,
        [0.01, 1.0],
        ensemble_size=2,
        seed=5,
    )
    assert 0.0 <= selection.scale <= 1.0
    assert rows
