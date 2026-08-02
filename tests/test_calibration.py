"""Gate 3 as unit tests: identifiability, recovery, and refusal to guess."""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.channel.analytic import passive_concentration, peak_time_passive
from agentjack.channel.calibrate import (
    calibration_report,
    fit_multi_distance,
    fit_passive_trace,
    shape_timescale,
    synthetic_trace,
)


def test_passive_peak_matches_closed_form():
    D, d = 79.4e-12, 20e-6
    t = np.linspace(1e-3, 20, 8000)
    observed = float(t[np.argmax(passive_concentration(t, D, d))])
    assert observed == pytest.approx(peak_time_passive(D, d), rel=0.01)


def test_the_model_is_degenerate_without_known_geometry():
    """Documents WHY d_known is required: scaling D by k, d by sqrt(k) and A by
    k^{3/2} leaves the curve bit-identical, so a free-geometry fit is meaningless.
    """
    t = np.linspace(0.05, 6, 400)
    base = passive_concentration(t, 79.4e-12, 20e-6, 0.0, 1e-16)
    for k in [10.0, 100.0, 1e4]:
        alt = passive_concentration(t, 79.4e-12 * k, 20e-6 * np.sqrt(k), 0.0, 1e-16 * k**1.5)
        assert np.abs(alt - base).max() / base.max() < 1e-12


def test_shape_timescale_is_the_invariant():
    assert shape_timescale(79.4e-12, 20e-6) == pytest.approx(
        shape_timescale(79.4e-11, 20e-6 * np.sqrt(10)))


def test_exact_recovery_without_noise():
    t, y, truth = synthetic_trace(noise_sigma=0.0, seed=1)
    f = fit_passive_trace(t, y, d_known=truth["d"])
    assert calibration_report(f, truth)["D_rel_error"] < 1e-3
    assert f.r_squared > 0.999


@pytest.mark.parametrize("noise", [0.01, 0.05, 0.10])
def test_recovery_degrades_gracefully_with_noise(noise):
    t, y, truth = synthetic_trace(noise_sigma=noise, seed=1)
    f = fit_passive_trace(t, y, d_known=truth["d"])
    assert calibration_report(f, truth)["D_rel_error"] < 0.05


def test_joint_fit_across_distances():
    traces = [(*synthetic_trace(d=d, noise_sigma=0.05, seed=int(d * 1e7))[:2], d)
              for d in [15e-6, 20e-6, 30e-6, 40e-6]]
    f = fit_multi_distance(traces)
    assert abs(f.D - 79.4e-12) / 79.4e-12 < 0.05


def test_joint_fit_needs_distinct_distances():
    tr = [(*synthetic_trace(d=20e-6, seed=i)[:2], 20e-6) for i in range(2)]
    with pytest.raises(ValueError, match="distances must differ"):
        fit_multi_distance(tr)
    with pytest.raises(ValueError, match="at least two"):
        fit_multi_distance(tr[:1])


def test_refuses_when_rising_edge_is_unresolved():
    """The pulse peaks before the first sample; argmax then finds the drift ramp."""
    t, y, truth = synthetic_trace(D=1e-7, noise_sigma=0.02, seed=5)
    with pytest.raises(ValueError, match="interior peak|rising edge"):
        fit_passive_trace(t, y, d_known=truth["d"])


def test_refuses_a_baseline_only_record():
    t = np.arange(0.1, 6.0, 0.1)
    y = 0.05 + 0.01 * t
    with pytest.raises(ValueError, match="interior peak"):
        fit_passive_trace(t, y, d_known=20e-6)


def test_requires_known_distance():
    t, y, _ = synthetic_trace(seed=0)
    with pytest.raises(TypeError):
        fit_passive_trace(t, y)
    with pytest.raises(ValueError):
        fit_passive_trace(t, y, d_known=-1.0)


def test_rejects_malformed_input():
    t, y, truth = synthetic_trace(seed=0)
    with pytest.raises(ValueError):
        fit_passive_trace(t, y[:-1], d_known=truth["d"])
    with pytest.raises(ValueError):
        fit_passive_trace(t[:5], y[:5], d_known=truth["d"])


@pytest.mark.parametrize("v", [1e-5, 5e-5])
def test_velocity_recovered_when_flow_dominates(v):
    t, y, truth = synthetic_trace(v=v, noise_sigma=0.02, seed=3)
    f = fit_passive_trace(t, y, d_known=truth["d"], fit_drift_velocity=True)
    assert calibration_report(f, truth)["v_rel_error"] < 0.05


def test_velocity_is_non_negative_by_construction():
    """Sign is not identifiable (only v^2 enters), so geometry fixes it."""
    t, y, truth = synthetic_trace(v=1e-5, noise_sigma=0.02, seed=3)
    f = fit_passive_trace(t, y, d_known=truth["d"], fit_drift_velocity=True)
    assert f.v >= 0.0


def test_baseline_dominated_pulse_is_refused():
    """A pulse dwarfed by sensor drift has no interior maximum, so it is
    rejected rather than fitted. Earlier this fitted with R^2 > 0.99 while
    recovering a D that was wrong by 99% - the variance being explained was
    the drift ramp, not the channel."""
    t, y, truth = synthetic_trace(offset=50.0, drift=5.0, noise_sigma=0.01, seed=2)
    with pytest.raises(ValueError, match="interior peak"):
        fit_passive_trace(t, y, d_known=truth["d"])


def test_report_exposes_pulse_to_baseline_ratio():
    t, y, truth = synthetic_trace(offset=0.05, drift=0.002, noise_sigma=0.01, seed=2)
    f = fit_passive_trace(t, y, d_known=truth["d"])
    assert calibration_report(f, truth)["pulse_to_baseline"] > 1.0


def test_missing_dataset_gives_instructions_not_a_traceback():
    from agentjack.data.loaders.mc_testbed import DataNotAvailable, load_macroscale_ethanol

    with pytest.raises(DataNotAvailable) as e:
        load_macroscale_ethanol(root="data/raw/definitely_not_here")
    assert "ieee-dataport" in str(e.value).lower()
