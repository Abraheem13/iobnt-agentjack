"""Closed-form references for the 3-D diffusive MC channel.

Ground truth for Gate 1. Nothing here may depend on the simulator; the whole
point is that these are derived independently so the simulator can be checked
against them.

Geometry: a point transmitter at distance ``d`` from the centre of a fully
absorbing spherical receiver of radius ``r_rx``, in an unbounded 3-D medium with
diffusion coefficient ``D``. Requires ``d > r_rx``.

First-passage (hitting) time density for a single molecule::

    f(t) = (r_rx / d) * (d - r_rx) / sqrt(4 pi D t^3)
           * exp( -(d - r_rx)^2 / (4 D t) )

Cumulative hitting probability::

    F(t) = (r_rx / d) * erfc( (d - r_rx) / sqrt(4 D t) )

As t -> inf, F -> r_rx / d: most molecules never arrive. That ratio is the
single most important number in the channel, and it caps every SNR you will
ever see.

All quantities are SI (metres, seconds). float64 only - see docs/05_local_dev.md.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

__all__ = [
    "hitting_rate",
    "cumulative_hitting_probability",
    "asymptotic_absorption",
    "characteristic_time",
    "discrete_cir",
    "passive_concentration",
    "peak_time_passive",
]


def _check_geometry(r_rx: float, d: float) -> None:
    if r_rx <= 0:
        raise ValueError(f"r_rx must be positive, got {r_rx}")
    if d <= r_rx:
        raise ValueError(f"transmitter must be outside the receiver: need d > r_rx, got d={d}, r_rx={r_rx}")


def hitting_rate(t: np.ndarray | float, D: float, r_rx: float, d: float) -> np.ndarray:
    """First-passage time density f(t). Zero for t <= 0."""
    _check_geometry(r_rx, d)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t > 0
    if not np.any(pos):
        return out
    tp = t[pos] if t.ndim else t
    delta = d - r_rx
    out_pos = (r_rx / d) * delta / np.sqrt(4.0 * np.pi * D * tp**3) * np.exp(-(delta**2) / (4.0 * D * tp))
    if t.ndim:
        out[pos] = out_pos
        return out
    return out_pos


def cumulative_hitting_probability(t: np.ndarray | float, D: float, r_rx: float, d: float) -> np.ndarray:
    """F(t): probability one molecule has been absorbed by time t."""
    _check_geometry(r_rx, d)
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t > 0
    if not np.any(pos):
        return out
    tp = t[pos] if t.ndim else t
    val = (r_rx / d) * erfc((d - r_rx) / np.sqrt(4.0 * D * tp))
    if t.ndim:
        out[pos] = val
        return out
    return val


def asymptotic_absorption(r_rx: float, d: float) -> float:
    """F(inf) = r_rx / d. The hard ceiling on received energy."""
    _check_geometry(r_rx, d)
    return r_rx / d


def characteristic_time(D: float, r_rx: float, d: float) -> float:
    """(d - r_rx)^2 / (4D): the timescale the symbol duration must respect.

    A symbol duration far below this gives a degenerate channel - almost nothing
    arrives in-slot and the ISI tail carries everything.
    """
    _check_geometry(r_rx, d)
    return (d - r_rx) ** 2 / (4.0 * D)


def discrete_cir(n_slots: int, T_s: float, D: float, r_rx: float, d: float) -> np.ndarray:
    """Per-slot absorption probabilities h[k] for k = 1..n_slots.

    h[k] is the probability a molecule released at t=0 is absorbed during slot k.
    h.sum() < r_rx/d always, with the deficit being the tail beyond n_slots.
    """
    if n_slots < 1:
        raise ValueError("n_slots must be >= 1")
    if T_s <= 0:
        raise ValueError("T_s must be positive")
    edges = np.arange(0, n_slots + 1, dtype=np.float64) * T_s
    F = cumulative_hitting_probability(edges, D, r_rx, d)
    return np.diff(F)


# ---------------------------------------------------------------------------
# Passive (transparent) observer
#
# The absorbing-receiver model above governs our simulated link. Real testbeds
# usually do NOT absorb: the macroscale ethanol platform reads concentration
# with a COTS alcohol sensor that leaves the molecules in the medium. Fitting an
# absorbing model to a passive measurement would bias every parameter, so
# calibration uses the model below instead. Getting this distinction wrong is a
# silent, systematic error, which is why it lives here explicitly.


def passive_concentration(
    t: np.ndarray | float,
    D: float,
    d: float,
    v: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Concentration at distance d from an impulsive point release.

        c(t) = A / (4 pi D t)^{3/2} * exp( -(d - v t)^2 / (4 D t) )

    ``v`` is bulk drift towards the receiver (0 for pure diffusion).
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t > 0
    if not np.any(pos):
        return out
    tp = t[pos] if t.ndim else t
    val = amplitude / (4.0 * np.pi * D * tp) ** 1.5 * np.exp(-((d - v * tp) ** 2) / (4.0 * D * tp))
    if t.ndim:
        out[pos] = val
        return out
    return val


def peak_time_passive(D: float, d: float) -> float:
    """Time of peak concentration for pure diffusion: d^2 / (6 D).

    Used to seed the calibration optimiser from the observed peak.
    """
    return d**2 / (6.0 * D)
