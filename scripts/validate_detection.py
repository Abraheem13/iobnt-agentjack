"""GATE 2: the receiver is good enough that attacks are not hitting a strawman.

  2a  DFE bit-error rate falls monotonically with molecule budget.
  2b  DFE clearly beats plain threshold detection (this channel is ISI-limited,
      so a fixed threshold cannot work - that is physics, not a bad baseline).
  2c  Frame success at the default operating point is high enough for the
      command pipeline to function at all. This is the check that matters:
      a 40-bit frame at 9.8% BER arrives intact 1.6% of the time, which would
      silently destroy every downstream experiment.
  2d  The learned detector actually learns.

    python scripts/validate_detection.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentjack.channel.diffusion import ChannelParams, DiffusionChannel  # noqa: E402
from agentjack.channel.isi import isi_severity  # noqa: E402
from agentjack.physical.detector import (  # noqa: E402
    DecisionFeedbackDetector,
    GRUDetector,
    ThresholdDetector,
    ber,
)
from agentjack.physical.modulation import PREAMBLE, SYMBOL_BUDGET  # noqa: E402

FRAME_BITS = len(PREAMBLE) + SYMBOL_BUDGET
MIN_FRAME_SUCCESS = 0.95
MIN_DFE_ADVANTAGE = 10.0   # x better than threshold at the operating point


def evaluate(N: int, n_bits: int, rng, epochs: int, train_gru: bool):
    p = ChannelParams(n_molecules=N)
    tb = rng.integers(0, 2, n_bits)
    ch = DiffusionChannel(p, seed=0)
    tr = ch.transmit(tb)
    vb = rng.integers(0, 2, n_bits)
    vr = DiffusionChannel(p, seed=1).transmit(vb)

    th = ThresholdDetector().fit(tr, tb)
    dfe = DecisionFeedbackDetector(ch.impulse_response(), N)
    out = {"threshold": ber(vb, th.detect(vr)), "dfe": ber(vb, dfe.detect(vr))}
    if train_gru:
        gru = GRUDetector(window=16, hidden_size=48).fit(tr, tb, epochs=epochs)
        out["gru"] = ber(vb, gru.detect(vr))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n_bits = 8_000 if args.quick else 30_000
    epochs = 8 if args.quick else 25
    budgets = [2000, 4000, 8000, 16000] if args.quick else [1000, 2000, 4000, 8000, 16000, 32000]

    rng = np.random.default_rng(0)
    p0 = ChannelParams()
    ch0 = DiffusionChannel(p0)

    print("GATE 2 :: detection\n")
    print(f"  ISI severity (sum h[1:] / h[0])   {isi_severity(ch0.impulse_response()):.3f}")
    print(f"  frame length                      {FRAME_BITS} bits "
          f"({len(PREAMBLE)} preamble + {SYMBOL_BUDGET} payload)")
    print(f"  default molecule budget           {p0.n_molecules}\n")

    rows = []
    print(f"  {'N':>7} {'threshold':>10} {'DFE':>10} {'GRU':>10} {'DFE frame ok':>13}")
    for N in budgets:
        r = evaluate(N, n_bits, rng, epochs, train_gru=True)
        rows.append((N, r))
        frame_ok = (1 - r["dfe"]) ** FRAME_BITS
        print(f"  {N:>7} {r['threshold']:>10.5f} {r['dfe']:>10.5f} {r['gru']:>10.5f} {frame_ok:>13.3f}")
    print()

    dfe_curve = [r["dfe"] for _, r in rows]
    gru_curve = [r["gru"] for _, r in rows]
    at_default = next((r for N, r in rows if N == p0.n_molecules), None)

    results = []

    ok = all(a >= b - 1e-12 for a, b in zip(dfe_curve, dfe_curve[1:], strict=False))
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 2a monotone DFE     BER falls with molecule budget")

    if at_default is None:
        print(f"[ FAIL ] 2b advantage         default budget {p0.n_molecules} not in sweep")
        results.append(False)
    else:
        adv = at_default["threshold"] / max(at_default["dfe"], 1e-9)
        ok = adv > MIN_DFE_ADVANTAGE
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] 2b DFE advantage    {adv:.0f}x better than threshold "
              f"(need >{MIN_DFE_ADVANTAGE:.0f}x)")

    if at_default is None:
        results.append(False)
    else:
        frame_ok = (1 - at_default["dfe"]) ** FRAME_BITS
        ok = frame_ok >= MIN_FRAME_SUCCESS
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] 2c frame success    {frame_ok:.3f} at N={p0.n_molecules} "
              f"(need >={MIN_FRAME_SUCCESS})")

    ok = gru_curve[0] > gru_curve[-1] and gru_curve[-1] < 0.1
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 2d GRU learns       BER {gru_curve[0]:.4f} -> {gru_curve[-1]:.4f}")

    print()
    print("  note: the threshold detector is deliberately flat across budgets. Raising N")
    print("  scales signal and interference together, so an ISI-limited channel gains")
    print("  nothing from more molecules. This is expected physics and is reported, not")
    print("  hidden - it is also why attack A3 has room to operate.")
    print()

    if all(results):
        print("GATE 2 PASSED. Day 3 (testbed calibration) is unblocked.")
        return 0
    print("GATE 2 FAILED. Fix detection before building the twin on top of it.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
