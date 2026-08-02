"""Inter-symbol interference: construction, removal and a severity knob.

The ISI ratio (tail energy over first-tap energy) is the single number that says
how hard this channel is. At the default parameters it is about 5, meaning five
times more of a symbol's energy lands in later slots than in its own. That is
the property attack A3 exploits, so it is instrumented rather than assumed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_isi_matrix", "apply_isi", "isi_severity", "subtract_isi"]


def build_isi_matrix(cir: np.ndarray, n_symbols: int) -> np.ndarray:
    """Lower-triangular Toeplitz channel matrix H with y = H @ s."""
    cir = np.asarray(cir, dtype=np.float64)
    H = np.zeros((n_symbols, n_symbols), dtype=np.float64)
    for lag, tap in enumerate(cir):
        if lag >= n_symbols:
            break
        idx = np.arange(n_symbols - lag)
        H[idx + lag, idx] = tap
    return H


def apply_isi(symbols: np.ndarray, cir: np.ndarray, n_molecules: float = 1.0) -> np.ndarray:
    """Noiseless received mean, truncated to the transmitted length."""
    symbols = np.asarray(symbols, dtype=np.float64)
    cir = np.asarray(cir, dtype=np.float64)
    return np.convolve(symbols * n_molecules, cir)[: len(symbols)]


def isi_severity(cir: np.ndarray) -> float:
    """sum(h[1:]) / h[0]. Zero means an ISI-free channel."""
    cir = np.asarray(cir, dtype=np.float64)
    if cir[0] <= 0:
        return float("inf")
    return float(cir[1:].sum() / cir[0])


def subtract_isi(
    observation: float,
    decided_history: np.ndarray,
    cir: np.ndarray,
    n_molecules: float,
) -> float:
    """Remove the ISI contributed by already-decided symbols.

    ``decided_history`` is ordered most-recent-first, so element j pairs with
    tap cir[j + 1].
    """
    taps = cir[1 : len(decided_history) + 1]
    return float(observation - n_molecules * np.dot(taps, decided_history[: len(taps)]))
