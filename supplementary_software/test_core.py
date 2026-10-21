from __future__ import annotations

import numpy as np
import pytest
import torch
from core import (
    PeriodicEncoder,
    assert_geometry_unseen,
    periodic_boundary_mask,
    random_block_mask,
)


def test_geometry_unseen_accepts_exact_zero_target_weight() -> None:
    weights = np.zeros((3, 8), dtype=np.float32)
    weights[:, [0, 1, 3]] = np.asarray(
        [[0.2, 0.3, 0.5], [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]],
        dtype=np.float32,
    )
    assert_geometry_unseen(weights, held_out_family=2)


def test_geometry_unseen_rejects_any_target_component() -> None:
    weights = np.zeros((1, 8), dtype=np.float32)
    weights[0, 0] = 0.999
    weights[0, 2] = 0.001
    with pytest.raises(ValueError, match="geometry leakage"):
        assert_geometry_unseen(weights, held_out_family=2)


def test_masks_preserve_shape_and_requested_fraction() -> None:
    x = torch.ones((4, 2, 16, 16, 16))
    generator = torch.Generator().manual_seed(7)
    for mask_fn in (random_block_mask, periodic_boundary_mask):
        if mask_fn is periodic_boundary_mask:
            masked, mask = mask_fn(x, 0.25, 1.0, generator)
        else:
            masked, mask = mask_fn(x, 0.25, generator)
        assert masked.shape == x.shape
        assert mask.shape == (4, 1, 16, 16, 16)
        assert torch.allclose(mask.float().mean(), torch.tensor(0.25))
        assert torch.count_nonzero(masked) < torch.count_nonzero(x)


def test_encoder_handles_16_and_32_grids() -> None:
    model = PeriodicEncoder(descriptor_dim=17)
    for grid in (16, 32):
        x = torch.randn(2, 2, grid, grid, grid)
        latent, reconstruction, topology = model(x)
        assert latent.shape == (2, 64)
        assert reconstruction.shape == x.shape
        assert topology.shape == (2, 17)
