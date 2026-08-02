"""GATE 1: the simulator's physics against closed-form theory.

Three independent checks, in increasing order of how much they can embarrass us:

  1a  Numerical integration of the hitting-rate density f(t) must reproduce the
      closed-form cumulative F(t). Catches algebra errors.
  1b  F(t) -> r_rx/d as t grows. Catches normalisation errors.
  1c  A Brownian particle simulation must reproduce the analytic per-slot CIR.
      Catches "the formula is elegant but the channel does not behave that way".

Nothing downstream is trustworthy until this passes.

    python scripts/validate_channel_physics.py
    python scripts/validate_channel_physics.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentjack.channel.analytic import (  # noqa: E402
    asymptotic_absorption,
    cumulative_hitting_probability,
    discrete_cir,
    hitting_rate,
)
from agentjack.channel.diffusion import ChannelParams, DiffusionChannel  # noqa: E402

TOL_INTEGRATION = 1e-3   # relative
TOL_ASYMPTOTE = 1e-2     # relative
MAX_ABS_Z = 4.0          # per-slot agreement, in binomial standard errors
MAX_TOTAL_BIAS = 0.03    # relative, on total absorbed mass


def check_integration(p: ChannelParams) -> tuple[bool, str]:
    t_max = 20 * p.T_s
    t = np.linspace(1e-6, t_max, 400_000)
    f = hitting_rate(t, p.D, p.r_rx, p.d_tx_rx)
    numeric = np.trapezoid(f, t)
    closed = float(cumulative_hitting_probability(t_max, p.D, p.r_rx, p.d_tx_rx))
    rel = abs(numeric - closed) / closed
    ok = rel < TOL_INTEGRATION
    return ok, f"integral(f) = {numeric:.8f}  vs  F(t) = {closed:.8f}   rel err {rel:.2e}"


def check_asymptote(p: ChannelParams) -> tuple[bool, str]:
    t_huge = 1e7 * p.T_s
    F = float(cumulative_hitting_probability(t_huge, p.D, p.r_rx, p.d_tx_rx))
    ceiling = asymptotic_absorption(p.r_rx, p.d_tx_rx)
    rel = abs(F - ceiling) / ceiling
    ok = rel < TOL_ASYMPTOTE
    return ok, f"F(t->inf) = {F:.6f}  vs  r_rx/d = {ceiling:.6f}   rel err {rel:.2e}"


def check_monte_carlo(p: ChannelParams, n_particles: int, steps_per_slot: int):
    """Compare in units of binomial standard error, not raw relative error.

    A slot holding probability mass q has relative standard error 1/sqrt(N*q).
    At the counts this channel produces that is 8-20%, so a fixed relative
    tolerance tests the particle budget rather than the physics. The z-score
    below tests the physics.
    """
    ch = DiffusionChannel(p)
    analytic = discrete_cir(p.isi_memory_L, p.T_s, p.D, p.r_rx, p.d_tx_rx)
    mc = ch.monte_carlo_cir(n_particles=n_particles, steps_per_slot=steps_per_slot)

    se = np.sqrt(analytic * (1.0 - analytic) / n_particles)
    z = (mc - analytic) / se
    worst_z = float(np.abs(z).max())

    total_bias = float((mc.sum() - analytic.sum()) / analytic.sum())
    ok = worst_z < MAX_ABS_Z and abs(total_bias) < MAX_TOTAL_BIAS
    msg = f"worst |z| = {worst_z:.2f} (limit {MAX_ABS_Z}), total-mass bias = {total_bias:+.2%} (limit +/-{MAX_TOTAL_BIAS:.0%})"
    return ok, msg, analytic, mc, z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer particles; smoke test only")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    n_particles = 8_000 if args.quick else 40_000
    steps_per_slot = 200 if args.quick else 400

    p = ChannelParams()
    print("GATE 1 :: channel physics validation\n")
    print("parameters")
    for k, v in p.diagnostics().items():
        print(f"  {k:<28} {v:.6g}")
    warnings = p.warn_if_degenerate()
    for w in warnings:
        print(f"  WARN: {w}")
    print()

    results = []

    ok, msg = check_integration(p)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 1a integration    {msg}")

    ok, msg = check_asymptote(p)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 1b asymptote      {msg}")

    ok, msg, analytic, mc, z = check_monte_carlo(p, n_particles, steps_per_slot)
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 1c monte carlo    {msg}")
    print()
    print("  slot      analytic     monte-carlo      rel err        z")
    for k, (a, m, zz) in enumerate(zip(analytic, mc, z, strict=True), start=1):
        r = abs(m - a) / a if a > 0 else float("nan")
        print(f"  {k:>4}    {a:>10.6f}    {m:>10.6f}    {r:>9.2%}   {zz:>+6.2f}")
    print()

    if not args.no_figure:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out = Path("results/figures/fig1_channel_validation.pdf")
            out.parent.mkdir(parents=True, exist_ok=True)
            slots = np.arange(1, len(analytic) + 1)
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
            ax[0].stem(slots, analytic, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="analytic")
            ax[0].plot(slots, mc, "C3x", markersize=8, label=f"particle sim (N={n_particles:,})")
            ax[0].set_xlabel("slot $k$"); ax[0].set_ylabel("$h[k]$")
            ax[0].set_title("Discrete channel impulse response"); ax[0].legend(frameon=False)
            t = np.linspace(1e-4, p.isi_memory_L * p.T_s, 2000)
            ax[1].plot(t, cumulative_hitting_probability(t, p.D, p.r_rx, p.d_tx_rx), "C0-")
            ax[1].axhline(asymptotic_absorption(p.r_rx, p.d_tx_rx), color="C7", ls="--",
                          label=r"$r_{rx}/d$")
            ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("$F(t)$")
            ax[1].set_title("Cumulative absorption"); ax[1].legend(frameon=False)
            fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
            print(f"figure written: {out}")
        except Exception as e:  # noqa: BLE001
            print(f"figure skipped: {e}")

    print()
    if all(results):
        print("GATE 1 PASSED. Day 2 (modulation, detection, ISI) is unblocked.")
        return 0
    print("GATE 1 FAILED. Do not proceed - fix the channel first.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
