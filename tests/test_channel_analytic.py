"""Gate 1 as unit tests, so CI keeps enforcing it after Day 1."""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.channel.analytic import (
    asymptotic_absorption,
    characteristic_time,
    cumulative_hitting_probability,
    discrete_cir,
    hitting_rate,
)
from agentjack.channel.diffusion import ChannelParams, DiffusionChannel

D, R_RX, D_TX = 79.4e-12, 5.0e-6, 20.0e-6


def test_geometry_is_validated():
    with pytest.raises(ValueError):
        cumulative_hitting_probability(1.0, D, r_rx=20e-6, d=5e-6)
    with pytest.raises(ValueError):
        cumulative_hitting_probability(1.0, D, r_rx=-1e-6, d=20e-6)


def test_density_integrates_to_cumulative():
    t = np.linspace(1e-6, 10.0, 400_000)
    numeric = np.trapezoid(hitting_rate(t, D, R_RX, D_TX), t)
    closed = float(cumulative_hitting_probability(10.0, D, R_RX, D_TX))
    assert abs(numeric - closed) / closed < 1e-3


def test_cumulative_is_monotone_and_bounded():
    t = np.linspace(1e-6, 500.0, 5000)
    F = cumulative_hitting_probability(t, D, R_RX, D_TX)
    assert np.all(np.diff(F) >= -1e-15)
    assert np.all(F <= asymptotic_absorption(R_RX, D_TX) + 1e-12)


def test_asymptote_matches_ratio():
    F = float(cumulative_hitting_probability(1e9, D, R_RX, D_TX))
    assert abs(F - R_RX / D_TX) / (R_RX / D_TX) < 1e-3


def test_zero_and_negative_time_give_zero():
    assert float(cumulative_hitting_probability(0.0, D, R_RX, D_TX)) == 0.0
    assert np.all(cumulative_hitting_probability(np.array([-1.0, 0.0]), D, R_RX, D_TX) == 0.0)


def test_cir_sums_below_ceiling():
    h = discrete_cir(8, 0.5, D, R_RX, D_TX)
    assert np.all(h > 0)
    assert h.sum() < asymptotic_absorption(R_RX, D_TX)


def test_cir_is_not_degenerate_at_default_params():
    """Regression guard: the original shipped T_s made the channel useless."""
    p = ChannelParams()
    assert p.warn_if_degenerate() == []
    assert p.diagnostics()["mean_counts_first_slot"] > 5


def test_characteristic_time_scaling():
    assert characteristic_time(2 * D, R_RX, D_TX) == pytest.approx(
        characteristic_time(D, R_RX, D_TX) / 2
    )


def test_expected_counts_respect_causality():
    ch = DiffusionChannel(ChannelParams())
    y = ch.expected_counts(np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert y[0] == 0.0 and y[1] == 0.0
    assert y[2] > 0 and y[3] > 0
    assert y[3] > y[4]


def test_transmit_is_seed_reproducible():
    s = np.array([1.0, 0.0, 1.0, 1.0])
    a = DiffusionChannel(ChannelParams(), seed=7).transmit(s)
    b = DiffusionChannel(ChannelParams(), seed=7).transmit(s)
    assert np.array_equal(a, b)


def test_monte_carlo_rejects_tunnelling_step_size():
    ch = DiffusionChannel(ChannelParams())
    with pytest.raises(ValueError, match="tunnel"):
        ch.monte_carlo_cir(n_particles=10, steps_per_slot=1)


@pytest.mark.slow
def test_monte_carlo_matches_analytic():
    """Agreement measured in binomial standard errors, not relative error.

    At these probabilities a slot's relative standard error is 8-20% even at
    20k particles, so a fixed relative tolerance would test the particle budget
    rather than the physics.
    """
    n = 20_000
    p = ChannelParams()
    ch = DiffusionChannel(p)
    analytic = discrete_cir(p.isi_memory_L, p.T_s, D, R_RX, D_TX)
    mc = ch.monte_carlo_cir(n_particles=n, steps_per_slot=400)
    se = np.sqrt(analytic * (1 - analytic) / n)
    z = (mc - analytic) / se
    assert np.abs(z).max() < 4.0
    assert abs(mc.sum() - analytic.sum()) / analytic.sum() < 0.03


@pytest.mark.slow
def test_bridge_correction_removes_absorption_bias():
    """Without the correction the particle sim systematically undercounts."""
    n = 20_000
    p = ChannelParams()
    ch = DiffusionChannel(p)
    analytic = discrete_cir(p.isi_memory_L, p.T_s, D, R_RX, D_TX).sum()
    naive = ch.monte_carlo_cir(n_particles=n, steps_per_slot=400, bridge_correction=False).sum()
    corrected = ch.monte_carlo_cir(n_particles=n, steps_per_slot=400, bridge_correction=True).sum()
    assert naive < analytic
    assert abs(corrected - analytic) < abs(naive - analytic)
