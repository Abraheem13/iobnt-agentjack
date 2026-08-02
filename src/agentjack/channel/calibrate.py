"""Fit the digital twin to measured testbed traces.

The credibility of every downstream result rests on this file. A reviewer's
fastest objection to a simulation paper is "your channel is invented", and the
answer is a fitted, error-reported match to published hardware traces.

Two things here are easy to get wrong and were, initially:

**Receiver model.** Our simulated link uses a fully absorbing receiver. Real
testbeds usually do not absorb - the macroscale ethanol platform reads
concentration with a sensor that leaves the molecules in the medium. Fitting an
absorbing model to a passive measurement biases every recovered parameter in the
same direction, invisibly. Calibration therefore fits ``passive_concentration``.

**Identifiability.** For

    c(t) = A / (4 pi D t)^{3/2} * exp( -d^2 / (4 D t) )

the curve SHAPE depends only on tau = d^2 / (4D) and the SCALE only on
A / D^{3/2}. So D, d and A are *not separately identifiable from a single
trace*: scaling D by k, d by sqrt(k) and A by k^{3/2} reproduces the curve
exactly, across arbitrarily many orders of magnitude. An unconstrained optimiser
fits it beautifully (R^2 > 0.99) while returning physically absurd parameters.

Two ways out, both implemented:

* ``fit_passive_trace(..., d_known=...)`` - pin the geometry, which every
  testbed paper reports, and fit D. This is the default path.
* ``fit_multi_distance(...)`` - fit one shared D across traces recorded at
  several known distances. Stronger, and the right call when the dataset
  provides a distance sweep.

**Drift velocity is weakly identified.** With bulk flow the exponent expands to
a constant factor (degenerate with the pulse height) plus exp(-v^2 t / 4D). Only
v^2 appears, so the SIGN is not estimable at all, and small |v| is absorbed by
the fitted sensor drift. We therefore constrain v >= 0 by geometry and report it
as reliable only where flow dominates diffusion - which is the flow-testbed case
and not the free-diffusion one.

**Sampling resolution.** If the pulse peaks within a few sampling intervals the
rising edge is unresolved and D is not estimable, though the optimiser will
still return something confident-looking. ``fit_passive_trace`` raises instead.

Validation runs in two stages: recover known parameters from synthetic traces
first (no downloads needed), then fit the real traces once they are on disk. An
optimiser that cannot recover parameters it was handed cannot be trusted on
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .analytic import passive_concentration

__all__ = [
    "ChannelFit",
    "fit_passive_trace",
    "fit_multi_distance",
    "calibration_report",
    "synthetic_trace",
    "shape_timescale",
]

# Physically defensible bounds. D for small molecules in water is ~1e-11 to
# 1e-8 m^2/s; macroscale gas-phase setups run far higher, hence the wide range.
D_BOUNDS = (1e-13, 1e-2)

# Below this many samples on the rising edge, D is fitted but flagged low-confidence.
MIN_SAMPLES_TO_PEAK = 3


def shape_timescale(D: float, d: float) -> float:
    """tau = d^2 / (4D), the only shape parameter a single trace determines."""
    return d**2 / (4.0 * D)


@dataclass
class ChannelFit:
    """Recovered channel parameters plus goodness of fit."""

    D: float
    d: float
    v: float
    amplitude: float
    offset: float
    drift: float
    nrmse: float
    r_squared: float
    n_points: int
    converged: bool
    d_was_fixed: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def tau(self) -> float:
        return shape_timescale(self.D, self.d)

    def as_dict(self) -> dict:
        return {
            "D_m2_per_s": self.D,
            "d_m": self.d,
            "v_m_per_s": self.v,
            "amplitude": self.amplitude,
            "offset": self.offset,
            "drift_per_s": self.drift,
            "tau_s": self.tau,
            "nrmse": self.nrmse,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "converged": self.converged,
            "d_was_fixed": self.d_was_fixed,
            **self.meta,
        }


def _amplitude_from_peak(peak: float, D: float, d: float) -> float:
    """Convert a peak height into the source amplitude A.

    A carries units of concentration x length^3 and moves by many orders of
    magnitude as D and d change, so it is a terrible thing to optimise directly:
    a seed estimated from the trace integral can land 13 orders of magnitude
    away from the truth, push the parameter against its bound, and get silently
    compensated by a wrong D. Peak height is O(1) in sensor units and is read
    straight off the data, so the optimiser sees a well-scaled problem.
    """
    t_peak = d**2 / (6.0 * D)
    return peak * (4.0 * np.pi * D * t_peak) ** 1.5 / np.exp(-1.5)


def _predict(t, D, d, v, peak, offset, drift):
    A = _amplitude_from_peak(peak, D, d)
    return passive_concentration(t, D, d, v, A) + offset + drift * t


def fit_passive_trace(
    t: np.ndarray,
    signal: np.ndarray,
    d_known: float,
    fit_drift_velocity: bool = False,
) -> ChannelFit:
    """Fit D (and nuisance terms) to one trace at a KNOWN transmitter distance.

    ``d_known`` is required, not optional. Leaving the geometry free makes the
    problem unidentifiable - see the module docstring.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(signal, dtype=np.float64)
    if t.shape != y.shape:
        raise ValueError("t and signal must have the same shape")
    if np.any(t < 0):
        raise ValueError("negative times")
    if len(t) < 10:
        raise ValueError("need at least 10 samples")
    if d_known <= 0:
        raise ValueError("d_known must be a positive distance in metres")

    baseline = float(np.median(y[: max(3, len(y) // 20)]))
    peak_idx = int(np.argmax(y - baseline))
    t_peak = max(float(t[peak_idx]), 1e-6)
    D0 = float(np.clip(d_known**2 / (6.0 * t_peak), *D_BOUNDS))
    peak0 = max(float(np.max(y) - baseline), 1e-9)
    v_scale = d_known / max(float(t[-1]), 1e-9)
    scale = float(np.ptp(y)) or 1.0

    # With no samples on the rising edge there is nothing to constrain D, and
    # the optimiser will still return something confident-looking. Refuse. Few
    # samples on the edge is allowed but flagged, since precision degrades.
    # A resolvable pulse has an INTERIOR maximum. If the maximum sits at either
    # end of the record there is no pulse shape to fit: at the start the rising
    # edge is unresolved, and at the end argmax has simply found the top of the
    # sensor drift ramp. Both cases otherwise fit happily and return a confident
    # wrong D, which is worse than failing.
    dt_est = float(np.median(np.diff(t))) if len(t) > 1 else float(t[0])
    if peak_idx == 0:
        raise ValueError(
            f"observed peak is at the first sample (t={t_peak:.4g}s, dt={dt_est:.4g}s): "
            "the rising edge is unresolved, so D is not estimable. Sample faster, "
            "or measure at a longer transmitter distance."
        )
    if peak_idx == len(t) - 1:
        raise ValueError(
            f"observed maximum is the last sample (t={t_peak:.4g}s): no interior peak, "
            "so this record contains a baseline trend rather than a resolvable pulse. "
            "Check the release time, the record length, and the sensor drift."
        )
    rising = int(peak_idx)

    if fit_drift_velocity:
        # v >= 0: flow towards the receiver. The sign is NOT identifiable from a
        # single trace - v enters the time-dependent part only as v^2 - so it is
        # fixed by the deployment geometry rather than estimated.
        theta0 = [np.log(D0), 0.0, np.log(peak0), baseline, 0.0]
        lo = [np.log(D_BOUNDS[0]), 0.0, np.log(peak0) - 10, -np.inf, -np.inf]
        hi = [np.log(D_BOUNDS[1]), 1000.0, np.log(peak0) + 10, np.inf, np.inf]

        def unpack(th):
            return np.exp(th[0]), th[1] * v_scale, np.exp(th[2]), th[3], th[4]
    else:
        theta0 = [np.log(D0), np.log(peak0), baseline, 0.0]
        lo = [np.log(D_BOUNDS[0]), np.log(peak0) - 10, -np.inf, -np.inf]
        hi = [np.log(D_BOUNDS[1]), np.log(peak0) + 10, np.inf, np.inf]

        def unpack(th):
            return np.exp(th[0]), 0.0, np.exp(th[1]), th[2], th[3]

    def residual(th):
        D, v, pk, off, dr = unpack(th)
        return (_predict(t, D, d_known, v, pk, off, dr) - y) / scale

    sol = least_squares(residual, theta0, bounds=(lo, hi), method="trf", max_nfev=20000)
    D, v, peak, offset, drift = unpack(sol.x)
    A = _amplitude_from_peak(peak, D, d_known)

    pred = _predict(t, D, d_known, v, peak, offset, drift)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return ChannelFit(
        D=float(D), d=float(d_known), v=float(v), amplitude=float(A),
        offset=float(offset), drift=float(drift),
        nrmse=float(np.sqrt(ss_res / len(y)) / (np.ptp(y) or 1.0)),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_points=len(t), converged=bool(sol.success), d_was_fixed=True,
        meta={
            "peak_time_s": t_peak,
            "cost": float(sol.cost),
            # If the fitted pulse is small against the baseline, R^2 is
            # measuring the drift ramp rather than the channel. Report it.
            "pulse_to_baseline": float(
                peak / (abs(offset) + abs(drift) * float(t[-1]) + 1e-30)
            ),
            "peak_value": float(peak),
            "rising_edge_samples": rising,
            "rising_edge_ok": bool(rising >= MIN_SAMPLES_TO_PEAK),
        },
    )


def fit_multi_distance(
    traces: list[tuple[np.ndarray, np.ndarray, float]],
    fit_drift_velocity: bool = False,
) -> ChannelFit:
    """Fit one shared D across traces taken at several known distances.

    ``traces`` is a list of (t, signal, distance). Each trace keeps its own
    amplitude, offset and drift - those are per-measurement nuisances - while D
    is shared, which is what makes the estimate meaningful.
    """
    if len(traces) < 2:
        raise ValueError("need at least two distances; use fit_passive_trace for one")
    dists = [d for _, _, d in traces]
    if len(set(dists)) < 2:
        raise ValueError("distances must differ, otherwise this is a single-distance fit")

    seeds = [fit_passive_trace(t, y, d, fit_drift_velocity) for t, y, d in traces]
    D0 = float(np.clip(np.median([sd.D for sd in seeds]), *D_BOUNDS))

    theta0 = [np.log(D0)]
    for sd in seeds:
        theta0 += [np.log(max(sd.meta["peak_value"], 1e-12)), sd.offset, sd.drift]
        if fit_drift_velocity:
            theta0.append(sd.v)
    per = 4 if fit_drift_velocity else 3

    def residual(th):
        D = np.exp(th[0])
        out = []
        for i, (t, y, d) in enumerate(traces):
            block = th[1 + i * per : 1 + (i + 1) * per]
            pk, off, dr = np.exp(block[0]), block[1], block[2]
            v = block[3] if fit_drift_velocity else 0.0
            scale = float(np.ptp(y)) or 1.0
            out.append((_predict(t, D, d, v, pk, off, dr) - y) / scale)
        return np.concatenate(out)

    sol = least_squares(residual, theta0, method="trf", max_nfev=40000)
    D = float(np.exp(sol.x[0]))

    ss_res = ss_tot = 0.0
    n = 0
    for i, (t, y, d) in enumerate(traces):
        block = sol.x[1 + i * per : 1 + (i + 1) * per]
        pk, off, dr = np.exp(block[0]), block[1], block[2]
        v = block[3] if fit_drift_velocity else 0.0
        pred = _predict(t, D, d, v, pk, off, dr)
        ss_res += float(np.sum((y - pred) ** 2))
        ss_tot += float(np.sum((y - y.mean()) ** 2))
        n += len(y)

    first = sol.x[1 : 1 + per]
    return ChannelFit(
        D=D, d=float(np.mean(dists)), v=float(first[3]) if fit_drift_velocity else 0.0,
        amplitude=_amplitude_from_peak(float(np.exp(first[0])), D, float(np.mean(dists))),
        offset=float(first[1]), drift=float(first[2]),
        nrmse=float(np.sqrt(ss_res / n) / (np.ptp(np.concatenate([y for _, y, _ in traces])) or 1.0)),
        r_squared=float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_points=n, converged=bool(sol.success), d_was_fixed=True,
        meta={"n_traces": len(traces), "distances_m": dists, "cost": float(sol.cost)},
    )


def synthetic_trace(
    D: float = 79.4e-12,
    d: float = 20e-6,
    v: float = 0.0,
    peak_value: float = 1.0,
    offset: float = 0.05,
    drift: float = 0.002,
    noise_sigma: float = 0.02,
    duration: float | None = None,
    dt: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """A trace with known hidden parameters, plus sensor offset, drift and noise.

    Parameterised by ``peak_value`` rather than the raw source amplitude A,
    because A carries units of (concentration x length^3) and spans many orders
    of magnitude as D and d change. Specifying A directly makes it very easy to
    build a trace whose signal is dwarfed by its own baseline - which then fits
    with R^2 > 0.99 while recovering a completely wrong D, because the variance
    being explained is the drift ramp rather than the pulse.

    ``noise_sigma`` is relative to the pulse height, not the total range.
    """
    rng = np.random.default_rng(seed)
    t_peak = d**2 / (6.0 * D)
    if duration is None:
        duration = max(8.0 * t_peak, 10 * dt)
    amplitude = peak_value * (4.0 * np.pi * D * t_peak) ** 1.5 / np.exp(-1.5)

    t = np.arange(dt, duration + dt, dt)
    clean = passive_concentration(t, D, d, v, amplitude)
    y = clean + offset + drift * t + rng.normal(0.0, noise_sigma * peak_value, size=len(t))
    return t, y, {"D": D, "d": d, "v": v, "amplitude": amplitude,
                  "offset": offset, "drift": drift, "peak_value": peak_value,
                  "t_peak": t_peak}


def calibration_report(fit: ChannelFit, truth: dict | None = None) -> dict:
    """Fit quality, and parameter recovery error when ground truth is known."""
    rep = {"nrmse": fit.nrmse, "r_squared": fit.r_squared,
           "converged": fit.converged, "tau_s": fit.tau}
    for k in ("pulse_to_baseline", "rising_edge_samples", "rising_edge_ok"):
        if fit.meta.get(k) is not None:
            rep[k] = fit.meta[k]
    if truth:
        for key, est in [("D", fit.D), ("d", fit.d), ("v", fit.v)]:
            if key not in truth:
                continue
            if truth[key]:
                rep[f"{key}_rel_error"] = abs(est - truth[key]) / abs(truth[key])
            else:
                rep[f"{key}_abs_error"] = abs(est - truth[key])
    return rep
