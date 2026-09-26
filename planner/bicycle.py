"""Kinematic bicycle at the rear axle, forward Euler.

State (x, y, psi, v, a, delta), control (jerk, steer_rate). Forward Euler at 0.1 s is what nuPlan's
plant integrates with, so poses predicted here are ones the simulated vehicle can reproduce.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

DT = 0.1  # [s]
WHEEL_BASE = 3.089  # [m] nuPlan Pacifica
NX, NU = 6, 2


def step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Euler step. x (6,), u (2,) -> (6,). No clipping, so linearize() stays exact."""
    px, py, psi, v, a, delta = x
    dt, L = DT, WHEEL_BASE
    return np.array(
        [
            px + dt * v * math.cos(psi),
            py + dt * v * math.sin(psi),
            psi + dt * v * math.tan(delta) / L,
            v + dt * a,
            a + dt * u[0],
            delta + dt * u[1],
        ]
    )


def linearize(X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Jacobians. X (N+1, 6), U (N, 2) -> A (N, 6, 6), B (N, 6, 2). U is unused: dynamics are affine in u."""
    dt, L = DT, WHEEL_BASE
    psi, v, delta = X[:-1, 2], X[:-1, 3], X[:-1, 5]
    N = psi.shape[0]
    A = np.tile(np.eye(NX), (N, 1, 1))
    A[:, 0, 2] = -dt * v * np.sin(psi)
    A[:, 0, 3] = dt * np.cos(psi)
    A[:, 1, 2] = dt * v * np.cos(psi)
    A[:, 1, 3] = dt * np.sin(psi)
    A[:, 2, 3] = dt * np.tan(delta) / L
    A[:, 2, 5] = dt * v / (L * np.cos(delta) ** 2)
    A[:, 3, 4] = dt
    B = np.zeros((N, NX, NU))
    B[:, 4, 0] = dt
    B[:, 5, 1] = dt
    return A, B
