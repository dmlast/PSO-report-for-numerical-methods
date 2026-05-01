from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
Objective = Callable[[Array], Array]


@dataclass(frozen=True)
class PSOConfig:
    n_particles: int = 35
    n_iter: int = 70
    omega: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    seed: int = 7
    velocity_scale: float = 0.15
    init_center: tuple[float, ...] | None = None
    init_spread: float = 1.0


def make_quadratic(kappa: float = 10.0) -> Objective:
    """Return f(x)=0.5*(x1^2+kappa*x2^2), a 1-strongly convex quadratic."""

    def objective(x: Array) -> Array:
        x = np.asarray(x)
        return 0.5 * (x[..., 0] ** 2 + kappa * x[..., 1] ** 2)

    return objective


def make_rastrigin(amplitude: float = 10.0) -> Objective:
    """Return the n-dimensional Rastrigin benchmark function."""

    def objective(x: Array) -> Array:
        x = np.asarray(x)
        n = x.shape[-1]
        return amplitude * n + np.sum(
            x**2 - amplitude * np.cos(2.0 * np.pi * x), axis=-1
        )

    return objective


def initialize_positions(
    rng: np.random.Generator,
    bounds: Array,
    n_particles: int,
    init_center: tuple[float, ...] | None = None,
    init_spread: float = 1.0,
) -> Array:
    low, high = bounds[:, 0], bounds[:, 1]
    dim = bounds.shape[0]
    if init_center is None:
        return rng.uniform(low, high, size=(n_particles, dim))

    center = np.asarray(init_center, dtype=float)
    if center.shape != (dim,):
        raise ValueError(f"init_center must have shape {(dim,)}, got {center.shape}")
    span = (high - low) * init_spread
    positions = center + rng.uniform(-0.5 * span, 0.5 * span, size=(n_particles, dim))
    return np.clip(positions, low, high)


def pso(objective: Objective, bounds: Array, config: PSOConfig) -> dict[str, Array]:
    """Run global-best PSO and return full history for analysis and plots."""
    rng = np.random.default_rng(config.seed)
    bounds = np.asarray(bounds, dtype=float)
    dim = bounds.shape[0]
    low, high = bounds[:, 0], bounds[:, 1]

    positions = initialize_positions(
        rng,
        bounds,
        config.n_particles,
        init_center=config.init_center,
        init_spread=config.init_spread,
    )
    velocities = (
        rng.uniform(-(high - low), high - low, size=(config.n_particles, dim))
        * config.velocity_scale
    )

    personal_best = positions.copy()
    personal_values = objective(personal_best)
    best_idx = int(np.argmin(personal_values))
    global_best = personal_best[best_idx].copy()
    global_value = float(personal_values[best_idx])

    position_history = [positions.copy()]
    velocity_history = [velocities.copy()]
    best_history = [global_best.copy()]
    value_history = [global_value]

    for _ in range(config.n_iter):
        r1 = rng.random(size=(config.n_particles, dim))
        r2 = rng.random(size=(config.n_particles, dim))
        cognitive = config.c1 * r1 * (personal_best - positions)
        social = config.c2 * r2 * (global_best - positions)
        velocities = config.omega * velocities + cognitive + social
        positions = np.clip(positions + velocities, low, high)

        values = objective(positions)
        improved = values < personal_values
        personal_best[improved] = positions[improved]
        personal_values[improved] = values[improved]

        best_idx = int(np.argmin(personal_values))
        if float(personal_values[best_idx]) < global_value:
            global_value = float(personal_values[best_idx])
            global_best = personal_best[best_idx].copy()

        position_history.append(positions.copy())
        velocity_history.append(velocities.copy())
        best_history.append(global_best.copy())
        value_history.append(global_value)

    return {
        "positions": np.array(position_history),
        "velocities": np.array(velocity_history),
        "best_positions": np.array(best_history),
        "best_values": np.array(value_history),
    }


def hitting_time(values: Array, eps: float = 1e-3, f_star: float = 0.0) -> int | None:
    gaps = np.asarray(values) - f_star
    idx = np.where(gaps <= eps)[0]
    return int(idx[0]) if len(idx) else None


def summarize_run(history: dict[str, Array], eps: float = 1e-3) -> dict[str, float | int | None]:
    values = history["best_values"]
    return {
        "final_f": float(values[-1]),
        "hit_eps": hitting_time(values, eps=eps),
        "best_x1": float(history["best_positions"][-1, 0]),
        "best_x2": float(history["best_positions"][-1, 1]),
    }
