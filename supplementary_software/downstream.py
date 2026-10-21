"""Few-shot downstream estimators and source-only topology-OOD gate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]


def _as_2d(y: np.ndarray) -> tuple[np.ndarray, bool]:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim != 2:
        raise ValueError(f"expected one- or two-dimensional targets, got {y.shape}")
    return y, False


def _restore_y(y: np.ndarray, was_1d: bool) -> np.ndarray:
    return y[:, 0] if was_1d else y


def ridge_pipeline(alphas: list[float]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))),
        ]
    )


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    model = ridge_pipeline(alphas)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def crossfit_base_residuals(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    y2, was_1d = _as_2d(y)
    predictions = np.empty_like(y2)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("cross-fitting requires at least two groups")
    for group in unique_groups:
        val = groups == group
        train = ~val
        model = ridge_pipeline(alphas)
        model.fit(x[train], y2[train])
        predictions[val] = np.asarray(model.predict(x[val])).reshape(val.sum(), -1)
    return _restore_y(y2 - predictions, was_1d)


@dataclass(frozen=True)
class ResidualSystemPrediction:
    base: np.ndarray
    residual_mean: np.ndarray
    residual_std_scalar: np.ndarray
    distance_norm: np.ndarray
    uncertainty_norm: np.ndarray


def _fit_residual_ensemble(
    x: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    x_predict: np.ndarray,
    alphas: list[float],
    ensemble_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual2, was_1d = _as_2d(residual)
    selector = ridge_pipeline(alphas)
    selector.fit(x, residual2)
    alpha_value = selector.named_steps["ridge"].alpha_
    if np.ndim(alpha_value):
        alpha = float(np.asarray(alpha_value).ravel()[0])
    else:
        alpha = float(alpha_value)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    predict_members: list[np.ndarray] = []
    train_members: list[np.ndarray] = []
    for _ in range(ensemble_size):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(x[sampled], residual2[sampled])
        predict_members.append(np.asarray(model.predict(x_predict)).reshape(len(x_predict), -1))
        train_members.append(np.asarray(model.predict(x)).reshape(len(x), -1))
    predicted = np.stack(predict_members)
    train_predicted = np.stack(train_members)
    mean = predicted.mean(axis=0)
    std_scalar = np.sqrt(np.mean(predicted.std(axis=0, ddof=1) ** 2, axis=1))
    train_std_scalar = np.sqrt(np.mean(train_predicted.std(axis=0, ddof=1) ** 2, axis=1))
    return _restore_y(mean, was_1d), std_scalar, train_std_scalar


def _fit_fusion_with_bootstrap_disagreement(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    x_predict: np.ndarray,
    alphas: list[float],
    ensemble_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the nested fusion predictor and estimate disagreement separately.

    The point prediction is the RidgeCV fit on every training row. Bootstrap
    members are used only for disagreement, so the ungated residual is exactly
    the difference between the full fusion predictor and the explicit base.
    """
    y2, was_1d = _as_2d(y)
    selector = ridge_pipeline(alphas)
    selector.fit(x, y2)
    fused = np.asarray(selector.predict(x_predict)).reshape(len(x_predict), -1)
    alpha_value = selector.named_steps["ridge"].alpha_
    if np.ndim(alpha_value):
        alpha = float(np.asarray(alpha_value).ravel()[0])
    else:
        alpha = float(alpha_value)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    predict_members: list[np.ndarray] = []
    train_members: list[np.ndarray] = []
    for _ in range(ensemble_size):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(x[sampled], y2[sampled])
        predict_members.append(np.asarray(model.predict(x_predict)).reshape(len(x_predict), -1))
        train_members.append(np.asarray(model.predict(x)).reshape(len(x), -1))
    predicted = np.stack(predict_members)
    train_predicted = np.stack(train_members)
    std_scalar = np.sqrt(np.mean(predicted.std(axis=0, ddof=1) ** 2, axis=1))
    train_std_scalar = np.sqrt(np.mean(train_predicted.std(axis=0, ddof=1) ** 2, axis=1))
    return _restore_y(fused, was_1d), std_scalar, train_std_scalar


def _cross_family_distance(
    z_train: np.ndarray,
    train_groups: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(z_train)
    train_scaled = scaler.transform(z_train)
    test_scaled = scaler.transform(z_test)
    dimensional_scale = np.sqrt(z_train.shape[1])
    train_distances = pairwise_distances(train_scaled) / dimensional_scale
    same_family = train_groups[:, None] == train_groups[None, :]
    train_distances[same_family] = np.inf
    finite_cross_family = np.isfinite(train_distances).any(axis=1)
    if not finite_cross_family.all():
        raise ValueError("each training row needs at least one other topology family")
    reference = np.min(train_distances, axis=1)
    test_distance = np.min(pairwise_distances(test_scaled, train_scaled), axis=1) / dimensional_scale
    return reference, test_distance


def fit_residual_system(
    explicit_train: np.ndarray,
    latent_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    explicit_test: np.ndarray,
    latent_test: np.ndarray,
    alphas: list[float],
    ensemble_size: int,
    seed: int,
) -> ResidualSystemPrediction:
    base_model = ridge_pipeline(alphas)
    base_model.fit(explicit_train, y_train)
    base = np.asarray(base_model.predict(explicit_test))
    fusion_x_train = np.concatenate([explicit_train, latent_train], axis=1)
    fusion_x_test = np.concatenate([explicit_test, latent_test], axis=1)
    fused, residual_std, train_residual_std = _fit_fusion_with_bootstrap_disagreement(
        fusion_x_train,
        y_train,
        groups_train,
        fusion_x_test,
        alphas,
        ensemble_size,
        seed,
    )
    distance_reference, distance_test = _cross_family_distance(
        latent_train,
        groups_train,
        latent_test,
    )
    distance_scale = max(float(np.quantile(distance_reference, 0.9)), 1e-8)
    uncertainty_scale = max(float(np.quantile(train_residual_std, 0.9)), 1e-8)
    return ResidualSystemPrediction(
        base=base,
        residual_mean=fused - base,
        residual_std_scalar=residual_std,
        distance_norm=distance_test / distance_scale,
        uncertainty_norm=residual_std / uncertainty_scale,
    )


@dataclass(frozen=True)
class GateSelection:
    scale: float
    lambda_distance: float
    lambda_uncertainty: float
    inner_macro_rmse: float
    inner_family_se: float
    base_macro_rmse: float
    ungated_macro_rmse: float
    source_negative_transfer_rate: float
    selected_mean_gate: float
    eligible_candidates: int


def gate_values(
    distance_norm: np.ndarray,
    uncertainty_norm: np.ndarray,
    scale: float,
    lambda_distance: float,
    lambda_uncertainty: float,
) -> np.ndarray:
    distance_excess = np.maximum(np.asarray(distance_norm) - 1.0, 0.0)
    uncertainty_excess = np.maximum(np.asarray(uncertainty_norm) - 1.0, 0.0)
    return scale * np.exp(
        -lambda_distance * distance_excess**2
        -lambda_uncertainty * uncertainty_excess**2
    )


def _family_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    true2, _ = _as_2d(y_true)
    pred2, _ = _as_2d(y_pred)
    return np.asarray(
        [
            np.sqrt(np.mean((pred2[groups == group] - true2[groups == group]) ** 2))
            for group in np.unique(groups)
        ],
        dtype=np.float64,
    )


def select_gate_source_only(
    explicit_source: np.ndarray,
    latent_source: np.ndarray,
    y_source: np.ndarray,
    groups_source: np.ndarray,
    alphas: list[float],
    ensemble_size: int,
    seed: int,
) -> tuple[GateSelection, list[dict[str, Any]]]:
    """Select attenuation without using held-out-family labels.

    Each source topology family is treated as a pseudo-target in turn. A
    non-inferiority constraint retains candidates within 1% of the ungated
    fusion macro RMSE, then minimizes source-family negative transfer.
    """
    rows: list[dict[str, Any]] = []
    for inner_index, group in enumerate(np.unique(groups_source)):
        val = groups_source == group
        train = ~val
        system = fit_residual_system(
            explicit_source[train],
            latent_source[train],
            y_source[train],
            groups_source[train],
            explicit_source[val],
            latent_source[val],
            alphas,
            ensemble_size,
            seed + 104729 * inner_index,
        )
        y_val2, _ = _as_2d(y_source[val])
        base2, _ = _as_2d(system.base)
        residual2, _ = _as_2d(system.residual_mean)
        for local_index in range(val.sum()):
            rows.append(
                {
                    "group": int(group),
                    "y": y_val2[local_index],
                    "base": base2[local_index],
                    "residual": residual2[local_index],
                    "distance_norm": float(system.distance_norm[local_index]),
                    "uncertainty_norm": float(system.uncertainty_norm[local_index]),
                }
            )
    y = np.stack([row["y"] for row in rows])
    base = np.stack([row["base"] for row in rows])
    residual = np.stack([row["residual"] for row in rows])
    groups = np.asarray([row["group"] for row in rows], dtype=np.int64)
    distance = np.asarray([row["distance_norm"] for row in rows])
    uncertainty = np.asarray([row["uncertainty_norm"] for row in rows])
    candidates: list[dict[str, Any]] = []
    base_family_scores = _family_rmse(y, base, groups)
    grid = (0.0, 0.25, 0.5, 1.0, 2.0)
    for scale in (0.0, 1.0):
        lambda_pairs = ((0.0, 0.0),) if scale == 0.0 else (
            (lambda_d, lambda_u) for lambda_d in grid for lambda_u in grid
        )
        for lambda_d, lambda_u in lambda_pairs:
            gate = gate_values(distance, uncertainty, scale, lambda_d, lambda_u)
            pred = base + gate[:, None] * residual
            family_scores = _family_rmse(y, pred, groups)
            candidates.append(
                {
                    "scale": scale,
                    "lambda_distance": lambda_d,
                    "lambda_uncertainty": lambda_u,
                    "macro_rmse": float(family_scores.mean()),
                    "family_se": float(family_scores.std(ddof=1) / np.sqrt(len(family_scores))),
                    "mean_gate": float(gate.mean()),
                    "negative_transfer_rate": float(
                        np.mean(family_scores > base_family_scores)
                    ),
                }
            )
    ungated = next(
        item
        for item in candidates
        if item["scale"] == 1.0
        and item["lambda_distance"] == 0.0
        and item["lambda_uncertainty"] == 0.0
    )
    threshold = 1.01 * ungated["macro_rmse"]
    base_candidate = next(item for item in candidates if item["scale"] == 0.0)
    if ungated["macro_rmse"] <= 1.01 * base_candidate["macro_rmse"]:
        eligible = [
            item
            for item in candidates
            if item["scale"] == 1.0 and item["macro_rmse"] <= threshold
        ]
    else:
        eligible = [
            item
            for item in candidates
            if item["macro_rmse"] <= 1.01 * base_candidate["macro_rmse"]
        ]
    selected = min(
        eligible,
        key=lambda item: (
            item["negative_transfer_rate"],
            item["macro_rmse"],
            item["mean_gate"],
        ),
    )
    selection = GateSelection(
        scale=float(selected["scale"]),
        lambda_distance=float(selected["lambda_distance"]),
        lambda_uncertainty=float(selected["lambda_uncertainty"]),
        inner_macro_rmse=float(selected["macro_rmse"]),
        inner_family_se=float(selected["family_se"]),
        base_macro_rmse=float(base_candidate["macro_rmse"]),
        ungated_macro_rmse=float(ungated["macro_rmse"]),
        source_negative_transfer_rate=float(selected["negative_transfer_rate"]),
        selected_mean_gate=float(selected["mean_gate"]),
        eligible_candidates=len(eligible),
    )
    serializable_rows = [
        {
            **{key: value for key, value in row.items() if key not in {"y", "base", "residual"}},
            "y": row["y"].tolist(),
            "base": row["base"].tolist(),
            "residual": row["residual"].tolist(),
        }
        for row in rows
    ]
    return selection, serializable_rows
