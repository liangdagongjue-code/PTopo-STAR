"""Strict geometry-unseen pretraining for the PTopo-STAR paper experiment."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

SOFTWARE_ROOT = Path(__file__).resolve().parent
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

from latticetopo_pilot.tpms_geometry import (  # noqa: E402
    PRIMITIVE_NAMES,
    interpolate_field,
    load_physical_samples,
    primitive_basis,
    topology_descriptors,
    volume_channels,
)

FloatArray = NDArray[np.float32]


class FullConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seeds: list[int]
    support_seed: int
    grid_size: int
    wall_thickness: float
    synthetic_pretrain_samples: int
    pretrain_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    topology_loss_weight: float
    periodic_consistency_weight: float
    mask_fraction: float
    boundary_mask_fraction: float
    threads: int
    mixed_precision: bool
    variants: list[str]
    few_shots: list[int]
    support_repeats: int
    minimum_test_samples: int
    ridge_alphas: list[float]
    residual_ensemble_size: int
    inner_residual_ensemble_size: int
    bootstrap_replicates: int

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, value: list[str]) -> list[str]:
        allowed = {"masked_only", "periodic_only", "topology_only", "ptopo_ssl"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown pretraining variants: {sorted(unknown)}")
        return value


def load_config(path: Path) -> FullConfig:
    return FullConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class GeometryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, volumes: np.ndarray, descriptors: np.ndarray):
        self.volumes = torch.from_numpy(volumes.astype(np.float32))
        self.descriptors = torch.from_numpy(descriptors.astype(np.float32))

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.volumes[index], self.descriptors[index]


class PeriodicEncoder(nn.Module):
    """Small 3-D encoder with identical capacity in every SSL ablation."""

    def __init__(self, descriptor_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(2, 12, 3, stride=2, padding=1, padding_mode="circular"),
            nn.GroupNorm(3, 12),
            nn.GELU(),
            nn.Conv3d(12, 24, 3, stride=2, padding=1, padding_mode="circular"),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv3d(24, 48, 3, stride=2, padding=1, padding_mode="circular"),
            nn.GroupNorm(6, 48),
            nn.GELU(),
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 2 * 2 * 2, 64),
            nn.GELU(),
        )
        self.decoder_seed = nn.Sequential(nn.Linear(64, 48 * 2 * 2 * 2), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(24, 12, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(12, 2, 4, stride=2, padding=1),
        )
        self.topology_head = nn.Sequential(
            nn.Linear(64, 48),
            nn.GELU(),
            nn.Linear(48, descriptor_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = functional.adaptive_avg_pool3d(features, output_size=(2, 2, 2))
        return self.to_latent(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        seed = self.decoder_seed(latent).view(-1, 48, 2, 2, 2)
        reconstruction = self.decoder(seed)
        if reconstruction.shape[-3:] != x.shape[-3:]:
            reconstruction = functional.interpolate(
                reconstruction,
                size=x.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        return latent, reconstruction, self.topology_head(latent)


def random_block_mask(
    x: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
    block: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, n, _, _ = x.shape
    if n % block:
        raise ValueError(f"grid size {n} must be divisible by mask block {block}")
    blocks_per_axis = n // block
    total_blocks = blocks_per_axis**3
    selected = max(1, round(total_blocks * fraction))
    scores = torch.rand((batch, total_blocks), generator=generator, device=x.device)
    selected_blocks = scores.topk(selected, dim=1).indices
    coarse_mask = torch.zeros((batch, total_blocks), dtype=torch.bool, device=x.device)
    coarse_mask.scatter_(1, selected_blocks, True)
    coarse_mask = coarse_mask.view(batch, 1, blocks_per_axis, blocks_per_axis, blocks_per_axis)
    mask = coarse_mask.repeat_interleave(block, dim=2)
    mask = mask.repeat_interleave(block, dim=3)
    mask = mask.repeat_interleave(block, dim=4)
    masked = x.masked_fill(mask.expand_as(x), 0.0)
    return masked, mask


def periodic_boundary_mask(
    x: torch.Tensor,
    fraction: float,
    boundary_fraction: float,
    generator: torch.Generator,
    block: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask spatially clustered blocks that can wrap across unit-cell faces."""
    batch, _, n, _, _ = x.shape
    if n % block:
        raise ValueError(f"grid size {n} must be divisible by mask block {block}")
    bpa = n // block
    selected = max(1, round((bpa**3) * fraction))
    side = max(1, int(np.ceil(selected ** (1.0 / 3.0))))
    offsets = torch.cartesian_prod(
        torch.arange(side, device=x.device),
        torch.arange(side, device=x.device),
        torch.arange(side, device=x.device),
    )
    if len(offsets) < selected:
        raise RuntimeError("periodic mask candidate construction failed")
    starts = torch.randint(0, bpa, (batch, 3), generator=generator, device=x.device)
    force_boundary = (
        torch.rand((batch,), generator=generator, device=x.device) < boundary_fraction
    )
    boundary_axes = torch.randint(0, 3, (batch,), generator=generator, device=x.device)
    boundary_rows = torch.nonzero(force_boundary, as_tuple=False).flatten()
    starts[boundary_rows, boundary_axes[boundary_rows]] = bpa - 1
    candidates = (starts[:, None, :] + offsets[None, :, :]) % bpa
    candidate_linear = (
        candidates[..., 0] * (bpa**2) + candidates[..., 1] * bpa + candidates[..., 2]
    )
    scores = torch.rand(
        (batch, len(offsets)), generator=generator, device=x.device
    )
    chosen = scores.topk(selected, dim=1).indices
    selected_linear = torch.gather(candidate_linear, 1, chosen)
    coarse_flat = torch.zeros((batch, bpa**3), dtype=torch.bool, device=x.device)
    coarse_flat.scatter_(1, selected_linear, True)
    coarse = coarse_flat.view(batch, 1, bpa, bpa, bpa)
    mask = coarse.repeat_interleave(block, dim=2)
    mask = mask.repeat_interleave(block, dim=3)
    mask = mask.repeat_interleave(block, dim=4)
    return x.masked_fill(mask.expand_as(x), 0.0), mask


@dataclass(frozen=True)
class VariantSpec:
    boundary_mask: bool
    periodic_consistency: bool
    topology_prediction: bool


VARIANT_SPECS = {
    "masked_only": VariantSpec(False, False, False),
    "periodic_only": VariantSpec(True, True, False),
    "topology_only": VariantSpec(False, False, True),
    "ptopo_ssl": VariantSpec(True, True, True),
}


def _cache_digest(config: FullConfig, held_out_family: int, seed: int) -> str:
    relevant = {
        "held_out_family": held_out_family,
        "seed": seed,
        "grid_size": config.grid_size,
        "wall_thickness": config.wall_thickness,
        "synthetic_pretrain_samples": config.synthetic_pretrain_samples,
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_geometry_unseen_cache(
    config: FullConfig,
    run_root: Path,
    held_out_family: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if held_out_family not in range(8):
        raise ValueError("held_out_family must be in [0, 7]")
    digest = _cache_digest(config, held_out_family, seed)
    cache_path = (
        run_root
        / "datasets"
        / "processed"
        / "ptopo_star_full"
        / f"geometry_fold{held_out_family}_seed{seed}_{digest}.npz"
    )
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            payload = {key: cached[key] for key in cached.files}
        assert_geometry_unseen(payload["pretrain_weights"], held_out_family)
        return payload

    rng = np.random.default_rng(seed + 1009 * held_out_family)
    basis = primitive_basis(config.grid_size)
    samples = load_physical_samples(
        run_root / "datasets" / "processed" / "tpms274_raw" / "dataset"
    )
    allowed = np.asarray([i for i in range(8) if i != held_out_family], dtype=np.int64)
    concentrations = np.asarray([0.12, 0.25, 0.5, 1.0, 2.0], dtype=np.float32)
    pretrain_weights: list[np.ndarray] = []
    thicknesses: list[float] = []
    for i in range(config.synthetic_pretrain_samples):
        weights = np.zeros(8, dtype=np.float32)
        if i < len(allowed):
            weights[allowed[i]] = 1.0
        else:
            alpha = float(rng.choice(concentrations))
            weights[allowed] = rng.dirichlet(np.full(len(allowed), alpha)).astype(np.float32)
        pretrain_weights.append(weights)
        thicknesses.append(float(rng.uniform(0.38, 0.62)))
    weights_array = np.stack(pretrain_weights)
    assert_geometry_unseen(weights_array, held_out_family)

    def render(weights: np.ndarray, thickness: float) -> tuple[np.ndarray, np.ndarray]:
        channels = volume_channels(interpolate_field(weights, basis), thickness)
        return channels.astype(np.float16), topology_descriptors(channels)

    pretrain_pairs = [
        render(weights, thickness)
        for weights, thickness in zip(weights_array, thicknesses, strict=True)
    ]
    target_pairs = [render(sample.weights, config.wall_thickness) for sample in samples]
    payload = {
        "pretrain_volumes": np.stack([item[0] for item in pretrain_pairs]),
        "pretrain_descriptors": np.stack([item[1] for item in pretrain_pairs]),
        "pretrain_weights": weights_array,
        "target_volumes": np.stack([item[0] for item in target_pairs]),
        "target_descriptors": np.stack([item[1] for item in target_pairs]),
        "target_weights": np.stack([sample.weights for sample in samples]),
        "target_energy": np.asarray(
            [sample.energy_dissipation for sample in samples], dtype=np.float32
        ),
        "target_peak": np.asarray([sample.peak_stress for sample in samples], dtype=np.float32),
        "target_curve": np.stack([sample.stress for sample in samples]),
        "target_strain": np.stack([sample.strain for sample in samples]),
        "target_groups": np.asarray(
            [sample.dominant_family for sample in samples], dtype=np.int64
        ),
        "target_ids": np.asarray([sample.sample_id for sample in samples], dtype="U128"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def assert_geometry_unseen(weights: np.ndarray, held_out_family: int) -> None:
    weights = np.asarray(weights)
    if weights.ndim != 2 or weights.shape[1] != 8:
        raise ValueError(f"expected [n, 8] pretraining weights, got {weights.shape}")
    if not np.allclose(weights[:, held_out_family], 0.0, atol=0.0):
        maximum = float(np.max(np.abs(weights[:, held_out_family])))
        raise ValueError(
            f"geometry leakage: family {held_out_family} has maximum pretraining weight {maximum}"
        )
    if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-6):
        raise ValueError("pretraining weights do not sum to one")


def train_variant(
    variant: str,
    config: FullConfig,
    volumes: np.ndarray,
    descriptors: np.ndarray,
    output_dir: Path,
    held_out_family: int,
    seed: int,
    device: torch.device,
) -> tuple[PeriodicEncoder, pd.DataFrame]:
    spec = VARIANT_SPECS[variant]
    random.seed(seed + 1009 * held_out_family)
    np.random.seed(seed + 1009 * held_out_family)
    torch.manual_seed(seed + 1009 * held_out_family)
    desc_mean = descriptors.mean(axis=0, keepdims=True)
    desc_std = descriptors.std(axis=0, keepdims=True) + 1e-6
    normalized = (descriptors - desc_mean) / desc_std
    dataset = GeometryDataset(volumes, normalized)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1009 * held_out_family),
        pin_memory=device.type == "cuda",
    )
    model = PeriodicEncoder(descriptors.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.pretrain_epochs,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    mask_generator = torch.Generator(device=device.type).manual_seed(
        seed + 1009 * held_out_family + 7919 * list(VARIANT_SPECS).index(variant)
    )
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, config.pretrain_epochs + 1):
        epoch_values: dict[str, list[float]] = {
            "loss": [],
            "reconstruction_loss": [],
            "topology_loss": [],
            "periodic_consistency_loss": [],
        }
        for volume, descriptor in loader:
            volume = volume.to(device, non_blocking=True)
            descriptor = descriptor.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                if spec.boundary_mask:
                    masked, mask = periodic_boundary_mask(
                        volume,
                        config.mask_fraction,
                        config.boundary_mask_fraction,
                        mask_generator,
                    )
                else:
                    masked, mask = random_block_mask(
                        volume,
                        config.mask_fraction,
                        mask_generator,
                    )
                latent, reconstruction, predicted_topology = model(masked)
                expanded_mask = mask.expand_as(volume)
                recon_loss = torch.mean(
                    (reconstruction[expanded_mask] - volume[expanded_mask]) ** 2
                )
                topology_loss = torch.zeros((), device=device)
                if spec.topology_prediction:
                    topology_loss = torch.mean((predicted_topology - descriptor) ** 2)
                periodic_loss = torch.zeros((), device=device)
                if spec.periodic_consistency:
                    shifts = tuple(
                        int(value)
                        for value in torch.randint(
                            -3,
                            4,
                            (3,),
                            generator=mask_generator,
                            device=device,
                        )
                    )
                    shifted_volume = torch.roll(volume, shifts=shifts, dims=(2, 3, 4))
                    shifted_mask = torch.roll(mask, shifts=shifts, dims=(2, 3, 4))
                    shifted_masked = shifted_volume.masked_fill(
                        shifted_mask.expand_as(volume), 0.0
                    )
                    shifted_latent = model.encode(shifted_masked)
                    periodic_loss = torch.mean((latent - shifted_latent) ** 2)
                loss = recon_loss
                if spec.topology_prediction:
                    loss = loss + config.topology_loss_weight * topology_loss
                if spec.periodic_consistency:
                    loss = loss + config.periodic_consistency_weight * periodic_loss
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_values["loss"].append(float(loss.detach()))
            epoch_values["reconstruction_loss"].append(float(recon_loss.detach()))
            epoch_values["topology_loss"].append(float(topology_loss.detach()))
            epoch_values["periodic_consistency_loss"].append(float(periodic_loss.detach()))
        scheduler.step()
        row = {
            "variant": variant,
            "held_out_family": held_out_family,
            "held_out_name": PRIMITIVE_NAMES[held_out_family],
            "seed": seed,
            "epoch": epoch,
            **{name: float(np.mean(values)) for name, values in epoch_values.items()},
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "descriptor_mean": desc_mean,
            "descriptor_std": desc_std,
            "config": config.model_dump(),
            "variant": variant,
            "held_out_family": held_out_family,
            "seed": seed,
        },
        output_dir / f"model_fold{held_out_family}_seed{seed}_{variant}.pt",
    )
    return model, pd.DataFrame(history)


def embed(
    model: PeriodicEncoder,
    volumes: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(volumes), batch_size):
            batch = torch.from_numpy(volumes[start : start + batch_size].astype(np.float32)).to(
                device
            )
            rows.append(model.encode(batch).cpu().numpy())
    return np.concatenate(rows, axis=0)
