"""GATE 3: the calibration fitter is trustworthy, then the twin matches hardware.

Stage A (always runs, no downloads):
  3a  Recover known D from synthetic traces across a noise sweep.
  3b  Recover a shared D jointly across several distances.
  3c  Refuse to fit when the rising edge is unresolved, rather than returning a
      confident wrong answer.

Stage B (runs only when Tier-A data is on disk):
  3d  Fit the real testbed traces and report NRMSE and R^2.

Stage A is not a warm-up. An optimiser that cannot recover parameters it was
handed cannot be trusted on hardware, and this one initially could not: the
model is degenerate in (D, d, A) unless the geometry is pinned.

    python scripts/validate_calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentjack.channel.calibrate import (  # noqa: E402
    calibration_report,
    fit_multi_distance,
    fit_passive_trace,
    synthetic_trace,
)
from agentjack.data.loaders.mc_testbed import DataNotAvailable, load_macroscale_ethanol  # noqa: E402

MAX_D_ERROR_CLEAN = 0.02
MAX_D_ERROR_NOISY = 0.05
MAX_D_ERROR_MULTI = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ethanol-distance", type=float, default=1.0,
                    help="reported TX-RX distance in metres for the ethanol testbed")
    args = ap.parse_args()

    noises = [0.0, 0.05] if args.quick else [0.0, 0.01, 0.02, 0.05, 0.10]
    results = []

    print("GATE 3 :: calibration\n")
    print("Stage A - parameter recovery on synthetic traces with known truth\n")
    print(f"  {'noise':>7} {'D rel err':>11} {'NRMSE':>8} {'R^2':>9} {'edge':>6} {'pulse/base':>11}")
    errs = []
    for ns in noises:
        t, y, truth = synthetic_trace(noise_sigma=ns, seed=1)
        f = fit_passive_trace(t, y, d_known=truth["d"])
        r = calibration_report(f, truth)
        errs.append(r["D_rel_error"])
        print(f"  {ns:>7.2f} {r['D_rel_error']:>11.5f} {f.nrmse:>8.4f} {f.r_squared:>9.5f} "
              f"{r['rising_edge_samples']:>6d} {r['pulse_to_baseline']:>11.1f}")
    print()

    ok = errs[0] < MAX_D_ERROR_CLEAN
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 3a clean recovery   D error {errs[0]:.5f} "
          f"(need <{MAX_D_ERROR_CLEAN})")

    ok = errs[-1] < MAX_D_ERROR_NOISY
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 3b noisy recovery   D error {errs[-1]:.5f} at "
          f"{noises[-1]:.0%} noise (need <{MAX_D_ERROR_NOISY})")

    traces = [(*synthetic_trace(d=d, noise_sigma=0.05, seed=int(d * 1e7))[:2], d)
              for d in [15e-6, 20e-6, 30e-6, 40e-6]]
    fm = fit_multi_distance(traces)
    err_multi = abs(fm.D - 79.4e-12) / 79.4e-12
    ok = err_multi < MAX_D_ERROR_MULTI
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 3c joint fit        D error {err_multi:.5f} across "
          f"{len(traces)} distances (need <{MAX_D_ERROR_MULTI})")

    refused = False
    try:
        t, y, truth = synthetic_trace(D=1e-7, noise_sigma=0.02, seed=5)
        fit_passive_trace(t, y, d_known=truth["d"])
    except ValueError:
        refused = True
    results.append(refused)
    print(f"[{'PASS' if refused else 'FAIL'}] 3d refuses garbage  unresolved rising edge is "
          f"rejected, not fitted")

    print("\nStage B - real testbed traces\n")
    try:
        ds = load_macroscale_ethanol(distance_m=args.ethanol_distance)
        t, y, d = ds.traces[0]
        f = fit_passive_trace(t, y, d_known=d)
        print(f"  dataset      {ds.name} ({len(ds)} traces)")
        print(f"  citation     {ds.citation}")
        print(f"  fitted D     {f.D:.4e} m^2/s   NRMSE {f.nrmse:.4f}   R^2 {f.r_squared:.5f}")
        out = Path("data/processed/calibration"); out.mkdir(parents=True, exist_ok=True)
        import json
        (out / "macroscale_ethanol.json").write_text(json.dumps(f.as_dict(), indent=2))
        print(f"  written      {out / 'macroscale_ethanol.json'}")
    except (DataNotAvailable, NotImplementedError) as e:
        print(f"  SKIPPED - {str(e).strip().splitlines()[0]}")
        print("  Stage A still gates the fitter. Re-run once the data is downloaded;")
        print("  the twin stays synthetic-only until then, and the paper says so.")

    print()
    if all(results):
        print("GATE 3 PASSED (Stage A). Day 4 (digital twin + actuation) is unblocked.")
        return 0
    print("GATE 3 FAILED. The fitter is not trustworthy - do not calibrate on real data yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
