"""Geometry utilities for the TPMS-274 topology-transfer pilot.

The eight implicit functions and their order follow the official TPMeSh
implementation used by the TPMS-274 dataset.  This pilot evaluates the fields
on a periodic grid; it does not redistribute TPMeSh or claim byte-identical
meshes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

PRIMITIVE_NAMES = (
    "Schoen Gyroid",
    "Schwarz Diamond",
    "Schwarz Primitive",
    "Schoen IWP",
    "Neovius",
    "Fischer-Koch S",
    "Schoen FRD",
    "PMY",
)


@dataclass(frozen=True)
class PhysicalSample:
    sample_id: str
    weights: FloatArray
    strain: FloatArray
    stress: FloatArray
    energy_dissipation: float
    peak_stress: float
    dominant_family: int


def energy_dissipation_from_curve(strain: FloatArray, stress: FloatArray) -> float:
    """Reproduce the TPMS-274 repository's hysteresis-loop objective.

    The official implementation splits the 120-point trace into equal loading
    and unloading halves, reverses unloading, and integrates their stress
    difference with a uniform step of ``max(strain) / 60``.  This is not
    equivalent to integrating the full non-uniform closed trace directly.
    """
    strain = np.asarray(strain, dtype=np.float32)
    stress = np.asarray(stress, dtype=np.float32)
    if strain.ndim != 1 or stress.ndim != 1 or strain.shape != stress.shape:
        raise ValueError("strain and stress must be one-dimensional arrays of equal length")
    if stress.size == 0 or stress.size % 2:
        raise ValueError("the stress-strain trace must contain a non-zero even number of points")
    half = stress.size // 2
    dx = float(np.max(strain)) / half
    loading = stress[:half]
    unloading = stress[half:][::-1]
    return float(np.trapezoid(loading - unloading, dx=dx))


def primitive_basis(grid_size: int) -> FloatArray:
    """Return the eight official TPMS implicit fields on [0, 2pi)^3."""
    axis = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    c2x, c2y, c2z = np.cos(2.0 * x), np.cos(2.0 * y), np.cos(2.0 * z)
    s2x, s2y, s2z = np.sin(2.0 * x), np.sin(2.0 * y), np.sin(2.0 * z)

    fields = np.stack(
        [
            sx * cy + sy * cz + sz * cx,
            cx * cy * cz - sx * sy * sz,
            cx + cy + cz,
            2.0 * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z),
            3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz,
            c2x * sy * cz + cx * c2y * sz + sx * cy * c2z,
            4.0 * cx * cy * cz - (c2x * c2y + c2y * c2z + c2z * c2x),
            2.0 * cx * cy * cz + s2x * sy + sx * s2z + s2y * sz,
        ],
        axis=0,
    )
    return fields.astype(np.float32)


def interpolate_field(weights: FloatArray, basis: FloatArray) -> FloatArray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (8,):
        raise ValueError(f"expected eight TPMS weights, got {weights.shape}")
    if not np.isclose(float(weights.sum()), 1.0, atol=2e-3):
        raise ValueError(f"TPMS weights must sum to one, got {weights.sum():.6f}")
    return np.tensordot(weights, basis, axes=(0, 0)).astype(np.float32)


def volume_channels(field: FloatArray, wall_thickness: float) -> FloatArray:
    """Build approximate signed-distance and wall-occupancy channels.

    TPMeSh defines a shell through |f| <= thickness/2 * ||grad f||.  Periodic
    central differences preserve the unit-cell boundary identification.
    """
    grid_size = field.shape[0]
    dx = 2.0 * np.pi / grid_size
    grads = [
        (np.roll(field, -1, axis=i) - np.roll(field, 1, axis=i)) / (2.0 * dx)
        for i in range(3)
    ]
    grad_norm = np.sqrt(sum(g * g for g in grads) + 1e-8)
    signed_distance = np.clip(field / grad_norm, -1.0, 1.0)
    wall = np.abs(field) <= (0.5 * wall_thickness * grad_norm)
    return np.stack([signed_distance, wall.astype(np.float32)], axis=0).astype(np.float32)


def periodic_euler(binary: NDArray[np.bool_]) -> float:
    """Euler characteristic of a voxel cubical complex on a 3-torus."""
    occ = np.asarray(binary, dtype=bool)
    cubes = int(occ.sum())
    faces = sum(int(np.logical_or(occ, np.roll(occ, 1, axis=a)).sum()) for a in range(3))

    edge_x = occ | np.roll(occ, 1, axis=1) | np.roll(occ, 1, axis=2) | np.roll(np.roll(occ, 1, axis=1), 1, axis=2)
    edge_y = occ | np.roll(occ, 1, axis=0) | np.roll(occ, 1, axis=2) | np.roll(np.roll(occ, 1, axis=0), 1, axis=2)
    edge_z = occ | np.roll(occ, 1, axis=0) | np.roll(occ, 1, axis=1) | np.roll(np.roll(occ, 1, axis=0), 1, axis=1)
    edges = int(edge_x.sum() + edge_y.sum() + edge_z.sum())

    vertices = np.zeros_like(occ)
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                vertices |= np.roll(occ, (sx, sy, sz), axis=(0, 1, 2))
    return float(int(vertices.sum()) - edges + faces - cubes)


def _chord_stats(binary: NDArray[np.bool_], axis: int) -> tuple[float, float]:
    """Periodic mean chord lengths of solid and void along one axis."""
    occ = np.asarray(binary, dtype=bool)
    solid_voxels = int(occ.sum())
    void_voxels = int(occ.size - solid_voxels)
    entering_solid = (~np.roll(occ, 1, axis=axis)) & occ
    entering_void = np.roll(occ, 1, axis=axis) & (~occ)
    solid_chords = int(entering_solid.sum())
    void_chords = int(entering_void.sum())
    full_solid = np.all(occ, axis=axis).sum()
    full_void = np.all(~occ, axis=axis).sum()
    solid_chords += int(full_solid)
    void_chords += int(full_void)
    n = occ.shape[axis]
    solid_mean = solid_voxels / max(solid_chords, 1) / n
    void_mean = void_voxels / max(void_chords, 1) / n
    if solid_voxels == 0:
        solid_mean = 0.0
    if void_voxels == 0:
        void_mean = 0.0
    return float(solid_mean), float(void_mean)


def topology_descriptors(channels: FloatArray) -> FloatArray:
    """Geometry-derived pseudo-labels used only during self-supervision."""
    signed_distance = channels[0]
    wall = channels[1] > 0.5
    n = wall.shape[0]
    transitions = [
        float(np.not_equal(wall, np.roll(wall, 1, axis=a)).mean()) for a in range(3)
    ]
    chords = [_chord_stats(wall, a) for a in range(3)]
    solid_chords = [x[0] for x in chords]
    void_chords = [x[1] for x in chords]
    grad_energy = [
        float(np.mean(np.abs(np.roll(signed_distance, -1, axis=a) - signed_distance)))
        for a in range(3)
    ]
    q10, q50, q90 = np.quantile(signed_distance, [0.1, 0.5, 0.9])
    values = [
        float(wall.mean()),
        *transitions,
        periodic_euler(wall) / (n**3),
        periodic_euler(~wall) / (n**3),
        *solid_chords,
        *void_chords,
        *grad_energy,
        float(q10),
        float(q50),
        float(q90),
    ]
    return np.asarray(values, dtype=np.float32)


def load_physical_samples(dataset_root: Path) -> list[PhysicalSample]:
    samples: list[PhysicalSample] = []
    for path in sorted(dataset_root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = np.asarray(payload["params"], dtype=np.float32)
        strain = np.asarray(payload["strain"], dtype=np.float32)
        stress = np.asarray(payload["stress"], dtype=np.float32)
        dissipation = float(payload["energy_dissipation"])
        reproduced = energy_dissipation_from_curve(strain, stress)
        if not np.isclose(dissipation, reproduced, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"energy_dissipation mismatch in {path}: stored={dissipation:.9g}, "
                f"reproduced={reproduced:.9g}"
            )
        samples.append(
            PhysicalSample(
                sample_id=path.stem,
                weights=weights,
                strain=strain,
                stress=stress,
                energy_dissipation=dissipation,
                peak_stress=float(np.max(stress)),
                dominant_family=int(np.argmax(weights)),
            )
        )
    if len(samples) != 274:
        raise ValueError(f"expected 274 physical samples, found {len(samples)}")
    return samples
