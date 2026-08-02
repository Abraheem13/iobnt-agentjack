"""3-D free-diffusion MC channel with a fully absorbing spherical receiver.

Two independent implementations live here on purpose:

* :meth:`DiffusionChannel.impulse_response` - fast, analytic, used everywhere.
* :meth:`DiffusionChannel.monte_carlo_cir` - slow, particle-based, used only to
  prove the analytic path is right (Gate 1).

If those two ever disagree beyond tolerance, every downstream number is suspect.

Physics stays float64 on the CPU. Do not move any of this to MPS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic import asymptotic_absorption, characteristic_time, discrete_cir

__all__ = ["ChannelParams", "DiffusionChannel"]


@dataclass(frozen=True)
class ChannelParams:
    """SI units throughout."""

    D: float = 79.4e-12       # m^2/s
    r_rx: float = 5.0e-6      # m
    d_tx_rx: float = 20.0e-6  # m
    T_s: float = 0.5          # s
    n_molecules: int = 8000   # released per '1' symbol (set for frame success, see Gate 2)
    isi_memory_L: int = 8
    background_rate: float = 0.0

    def diagnostics(self) -> dict:
        h = discrete_cir(self.isi_memory_L, self.T_s, self.D, self.r_rx, self.d_tx_rx)
        ceiling = asymptotic_absorption(self.r_rx, self.d_tx_rx)
        return {
            "asymptotic_absorption": ceiling,
            "characteristic_time_s": characteristic_time(self.D, self.r_rx, self.d_tx_rx),
            "T_s_over_characteristic": self.T_s / characteristic_time(self.D, self.r_rx, self.d_tx_rx),
            "h_first_slot": float(h[0]),
            "captured_in_memory": float(h.sum()),
            "fraction_of_ceiling": float(h.sum() / ceiling),
            "mean_counts_first_slot": float(h[0] * self.n_molecules),
            "isi_ratio": float(h[1:].sum() / h[0]) if h[0] > 0 else np.inf,
        }

    def warn_if_degenerate(self) -> list[str]:
        """Catch parameter sets that silently produce a useless channel."""
        d = self.diagnostics()
        warnings = []
        if d["mean_counts_first_slot"] < 5:
            warnings.append(
                f"first-slot mean count is {d['mean_counts_first_slot']:.2f} molecules - "
                "counting noise will dominate; raise T_s or n_molecules"
            )
        if d["fraction_of_ceiling"] < 0.25:
            warnings.append(
                f"only {d['fraction_of_ceiling']:.1%} of reachable molecules land inside the "
                f"{self.isi_memory_L}-slot memory - the truncation is losing the signal"
            )
        if d["T_s_over_characteristic"] < 0.1:
            warnings.append(
                f"T_s is {d['T_s_over_characteristic']:.3f}x the characteristic diffusion time - "
                "the channel is pure ISI tail"
            )
        return warnings


class DiffusionChannel:
    """Discrete-time diffusive MC channel."""

    def __init__(self, params: ChannelParams | None = None, seed: int = 0):
        self.p = params or ChannelParams()
        self.rng = np.random.default_rng(seed)
        self._cir: np.ndarray | None = None

    def impulse_response(self, n_slots: int | None = None) -> np.ndarray:
        """h[k], k = 1..L. Cached; float64."""
        n = n_slots if n_slots is not None else self.p.isi_memory_L
        if self._cir is None or len(self._cir) != n:
            self._cir = discrete_cir(n, self.p.T_s, self.p.D, self.p.r_rx, self.p.d_tx_rx)
        return self._cir

    def expected_counts(self, symbols: np.ndarray) -> np.ndarray:
        """Noiseless mean molecule count per slot, including the ISI tail."""
        symbols = np.asarray(symbols, dtype=np.float64)
        h = self.impulse_response()
        released = symbols * self.p.n_molecules
        conv = np.convolve(released, h)[: len(symbols)]
        return conv + self.p.background_rate

    def transmit(self, symbols: np.ndarray) -> np.ndarray:
        """Observed counts per slot: Poisson around the expected counts."""
        return self.rng.poisson(self.expected_counts(symbols)).astype(np.float64)

    def monte_carlo_cir(
        self,
        n_slots: int | None = None,
        n_particles: int = 20000,
        steps_per_slot: int = 400,
        bridge_correction: bool = True,
        seed: int = 12345,
    ) -> np.ndarray:
        """Brownian particle simulation. VALIDATION ONLY - never in the hot path.

        Naive discrete-time stepping against an absorbing boundary undercounts
        absorption: a particle can cross the surface and return inside a single
        step, and the endpoint test never sees it. The bias is order sqrt(dt) and
        is large enough to matter (~7% at 400 steps/slot here).

        With ``bridge_correction`` we apply the standard Brownian-bridge hitting
        probability for a locally planar absorber (Andrews & Bray, 2004): for a
        step from surface distance ``a`` to surface distance ``b``, both outside,

            P(hit during step) = exp( -a * b / (D * dt) )

        valid while the step size is small against the radius of curvature, which
        the tunnelling guard below enforces.
        """
        n = n_slots if n_slots is not None else self.p.isi_memory_L
        dt = self.p.T_s / steps_per_slot
        sigma = np.sqrt(2.0 * self.p.D * dt)

        step_ratio = sigma / self.p.r_rx
        if step_ratio > 0.25:
            raise ValueError(
                f"step size {sigma:.2e} m is {step_ratio:.2f}x the receiver radius; "
                "particles will tunnel through the absorber. Raise steps_per_slot."
            )

        rng = np.random.default_rng(seed)
        pos = np.zeros((n_particles, 3), dtype=np.float64)
        rx_centre = np.array([self.p.d_tx_rx, 0.0, 0.0])
        alive = np.ones(n_particles, dtype=bool)
        absorbed_slot = np.full(n_particles, -1, dtype=np.int64)

        prev_gap = np.full(n_particles, self.p.d_tx_rx - self.p.r_rx, dtype=np.float64)

        for slot in range(n):
            for _ in range(steps_per_slot):
                idx = np.flatnonzero(alive)
                if idx.size == 0:
                    break
                pos[idx] += rng.normal(0.0, sigma, size=(idx.size, 3))
                gap = np.linalg.norm(pos[idx] - rx_centre, axis=1) - self.p.r_rx

                hit = gap <= 0.0
                if bridge_correction:
                    outside = ~hit
                    if np.any(outside):
                        a = prev_gap[idx][outside]
                        b = gap[outside]
                        p_bridge = np.exp(-a * b / (self.p.D * dt))
                        crossed = rng.random(a.size) < p_bridge
                        oi = np.flatnonzero(outside)
                        hit[oi[crossed]] = True

                hit_idx = idx[hit]
                if hit_idx.size:
                    alive[hit_idx] = False
                    absorbed_slot[hit_idx] = slot
                prev_gap[idx] = gap

        counts = np.bincount(absorbed_slot[absorbed_slot >= 0], minlength=n)[:n]
        return counts.astype(np.float64) / n_particles
