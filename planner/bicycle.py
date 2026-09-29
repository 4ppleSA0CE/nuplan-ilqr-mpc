"""Kinematic bicycle at the rear axle in path-curvature form, forward Euler.

State (x, y, psi, v, a, kappa), control (jerk, sigma), where kappa is the path curvature and sigma = d kappa / ds its
rate per metre travelled. Steering can therefore only change while the car moves. That matches nuPlan's LQR tracker,
which re-derives curvature from the planned poses alone: a plan that turned the wheels while standing still could
not be executed (P3 found the car stuck in tight turns for exactly that reason). Forward Euler at 0.1 s is what
nuPlan's plant integrates with, so poses predicted here are ones the simulated vehicle can reproduce.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

DT = 0.1  # [s]
WHEEL_BASE = 3.089  # [m] nuPlan Pacifica
NX, NU = 6, 2


def kappa_from_steering(delta: float) -> float:
    """Tire steering angle [rad] -> path curvature at the rear axle [1/m]."""
    return math.tan(delta) / WHEEL_BASE


def steering_from_kappa(kappa: float) -> float:
    return math.atan(kappa * WHEEL_BASE)


def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Euler step. x (6,), u (2,) -> (6,). No clipping, so linearize() stays exact."""
    px, py, psi, v, a, kappa = x
    dt = DT
    return np.array(
        [
            px + dt * v * math.cos(psi),
            py + dt * v * math.sin(psi),
            psi + dt * v * kappa,
            v + dt * a,
            a + dt * u[0],
            kappa + dt * v * u[1],
        ]
    )


def linearize(X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Jacobians. X (N+1, 6), U (N, 2) -> A (N, 6, 6), B (N, 6, 2)."""
    dt = DT
    psi, v, kappa = X[:-1, 2], X[:-1, 3], X[:-1, 5]
    N = psi.shape[0]
    A = np.tile(np.eye(NX), (N, 1, 1))
    A[:, 0, 2] = -dt * v * np.sin(psi)
    A[:, 0, 3] = dt * np.cos(psi)
    A[:, 1, 2] = dt * v * np.cos(psi)
    A[:, 1, 3] = dt * np.sin(psi)
    A[:, 2, 3] = dt * kappa
    A[:, 2, 5] = dt * v
    A[:, 3, 4] = dt
    A[:, 5, 3] = dt * U[:, 1]  # kappa+ = kappa + dt v sigma: the only place U enters A
    B = np.zeros((N, NX, NU))
    B[:, 4, 0] = dt
    B[:, 5, 1] = dt * v
    return A, B
