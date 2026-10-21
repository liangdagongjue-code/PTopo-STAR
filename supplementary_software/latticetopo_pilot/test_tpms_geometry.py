from __future__ import annotations

import numpy as np
from latticetopo_pilot.tpms_geometry import (
    energy_dissipation_from_curve,
    interpolate_field,
    periodic_euler,
    primitive_basis,
    topology_descriptors,
    volume_channels,
)


def test_basis_and_interpolation_shapes() -> None:
    basis = primitive_basis(16)
    field = interpolate_field(np.eye(8, dtype=np.float32)[0], basis)
    assert basis.shape == (8, 16, 16, 16)
    assert field.shape == (16, 16, 16)
    assert np.isfinite(field).all()


def test_periodic_euler_invariant_to_translation() -> None:
    rng = np.random.default_rng(5)
    binary = rng.random((8, 8, 8)) > 0.65
    assert periodic_euler(binary) == periodic_euler(np.roll(binary, (2, -1, 3), axis=(0, 1, 2)))


def test_topology_descriptors_translation_invariant() -> None:
    basis = primitive_basis(16)
    channels = volume_channels(interpolate_field(np.eye(8, dtype=np.float32)[4], basis), 0.5)
    shifted = np.roll(channels, (0, 3, -2, 1), axis=(0, 1, 2, 3))
    np.testing.assert_allclose(topology_descriptors(channels), topology_descriptors(shifted), atol=1e-6)


def test_energy_dissipation_uses_loading_unloading_difference() -> None:
    strain = np.asarray([0.0, 0.5, 1.0, 1.0, 0.5, 0.0], dtype=np.float32)
    stress = np.asarray([0.0, 1.0, 2.0, 1.0, 0.5, 0.0], dtype=np.float32)
    expected = np.trapezoid(
        stress[:3] - stress[3:][::-1],
        dx=float(strain.max()) / 3,
    )
    assert energy_dissipation_from_curve(strain, stress) == expected


def test_energy_dissipation_rejects_odd_trace() -> None:
    strain = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    stress = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    try:
        energy_dissipation_from_curve(strain, stress)
    except ValueError as exc:
        assert "even number" in str(exc)
    else:
        raise AssertionError("odd-length trace should be rejected")
