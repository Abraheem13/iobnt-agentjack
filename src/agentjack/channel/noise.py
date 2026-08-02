"""Counting noise, sensor noise and background interference.

Counting noise is Poisson and physical: it comes from molecules being discrete.
Sensor noise is additive and instrumental, calibrated on Day 3 against the
macroscale ethanol testbed, which used a COTS alcohol sensor with visible drift.
"""

from __future__ import annotations

import numpy as np

__all__ = ["poisson_counting", "sensor_noise", "background_interference"]


def poisson_counting(mean_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Discrete-molecule shot noise."""
    mean_counts = np.asarray(mean_counts, dtype=np.float64)
    if np.any(mean_counts < 0):
        raise ValueError("mean counts must be non-negative")
    return rng.poisson(mean_counts).astype(np.float64)


def sensor_noise(
    trace: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.0,
    drift_per_slot: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Additive Gaussian noise plus linear baseline drift and a fixed offset."""
    trace = np.asarray(trace, dtype=np.float64)
    n = len(trace)
    out = trace + offset + drift_per_slot * np.arange(n, dtype=np.float64)
    if sigma > 0:
        out = out + rng.normal(0.0, sigma, size=n)
    return out


def background_interference(rate: float, n_slots: int, rng: np.random.Generator) -> np.ndarray:
    """Molecules from sources other than the intended transmitter."""
    if rate < 0:
        raise ValueError("rate must be non-negative")
    return rng.poisson(rate, size=n_slots).astype(np.float64)
