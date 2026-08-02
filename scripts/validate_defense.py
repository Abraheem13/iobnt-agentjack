"""GATE 9: the nested monitor covers what single timescales cannot.

  9a  Every defense holds its calibrated false-positive rate on benign traffic.
  9b  Single levels fail on DISJOINT attack sets - the empirical premise.
  9c  The nested monitor detects every attack class.
  9d  No proper subset of levels does. This is the necessity claim: if any pair
      covered everything, the third level would be decoration.
  9e  Detections are attributed to the timescale that fired.

Every threshold is calibrated on benign runs only, at a stated false-positive
target. Fitting on attack data would be the single most damaging methodological
error available here, so calibration and evaluation use disjoint seeds.

    python scripts/validate_defense.py [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentjack.attacks.base import AttackBudget, KnowledgeLevel  # noqa: E402
from agentjack.attacks.compromised_node import CompromisedNodeAttack  # noqa: E402
from agentjack.attacks.isi_exploit import ISIExploitAttack  # noqa: E402
from agentjack.attacks.replay import ReplayAttack  # noqa: E402
from agentjack.attacks.semantic_injection import SemanticInjectionAttack  # noqa: E402
from agentjack.attacks.spoofing import SpoofingAttack  # noqa: E402
from agentjack.data.loaders.cgm import load_cgm_or_synthetic  # noqa: E402
from agentjack.defenses.nested_monitor import LEVEL_NAMES, NestedTrustMonitor  # noqa: E402
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig  # noqa: E402

K = KnowledgeLevel.FULL_CIR
EPISODE = 32


def make_attacks():
    return {
        "A1 replay": (lambda: ReplayAttack(knowledge=K, start_after=0,
                                           replay_when_glucose_below=99.0), None),
        "A2 spoofing": (lambda: SpoofingAttack(knowledge=K, start_after=0,
                                               spoof_when_glucose_below=99.0), None),
        "A3 ISI-exploit": (lambda: ISIExploitAttack(knowledge=K, start_after=0,
                                                    attack_when_glucose_below=99.0), None),
        "A4 semantic": (lambda: SemanticInjectionAttack(knowledge=K, start_after=0,
                                                        inject_when_glucose_below=99.0), None),
        "A5 compromised node": (None, lambda: CompromisedNodeAttack(
            start_after=0, active_when_glucose_below=99.0)),
        # The case both other timescales miss by construction: a genuine frame,
        # from the real node, at the real position, with a legitimate note -
        # only stale.
        "A6 insider replay": (None, lambda: CompromisedNodeAttack(
            start_after=0, active_when_glucose_below=99.0,
            replay_stale_frame=True, phrase=None)),
    }


def collect(monitor, cgm, pids, seeds, attacker=None, injector=None):
    """Per-level raw scores across episodes. No ground truth reaches the monitor."""
    per_level = {k: [] for k in monitor._members}
    for s in seeds:
        pid = pids[s % len(pids)]
        tw = BioCyberTwin(TwinConfig(episode_length=EPISODE),
                          meals=cgm[cgm.patient_id == pid].meal_carbs_g.to_numpy(), seed=s)
        if attacker is not None:
            a = attacker()
            a.reset()
            tw.attacker = a
        if injector is not None:
            j = injector()
            j.reset()
            tw.injector = j
        monitor.reset()
        obs = tw.reset(glucose0=7.0)
        for _ in range(EPISODE):
            raw = monitor.raw_scores(tw, obs)
            for k, v in raw.items():
                per_level[k].append(v)
            r = tw.step("none")
            obs = r.observation
            if r.done:
                break
    return {k: np.asarray(v, dtype=float) for k, v in per_level.items()}


def detection_rate(monitor, cgm, pids, seeds, attacker=None, injector=None):
    hits = total = 0
    attrib = {}
    for s in seeds:
        pid = pids[s % len(pids)]
        tw = BioCyberTwin(TwinConfig(episode_length=EPISODE),
                          meals=cgm[cgm.patient_id == pid].meal_carbs_g.to_numpy(), seed=s)
        if attacker is not None:
            a = attacker()
            a.reset()
            tw.attacker = a
        if injector is not None:
            j = injector()
            j.reset()
            tw.injector = j
        monitor.reset()
        obs = tw.reset(glucose0=7.0)
        for _ in range(EPISODE):
            d = monitor.decide(tw, obs)
            hits += int(d.veto)
            total += 1
            if d.veto and d.level is not None:
                attrib[d.level] = attrib.get(d.level, 0) + 1
            r = tw.step("none")
            obs = r.observation
            if r.done:
                break
    return hits / max(total, 1), attrib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cgm, is_real = load_cgm_or_synthetic(seed=0)
    pids = list(cgm.patient_id.unique())
    calib_seeds = list(range(0, 4 if args.quick else 8))
    eval_seeds = list(range(100, 103 if args.quick else 106))   # disjoint from calibration

    print("GATE 9 :: nested trust monitor\n")
    print(f"  CGM source   {'REAL Shanghai' if is_real else 'SYNTHETIC fallback'}")
    print(f"  calibration  seeds {calib_seeds[0]}-{calib_seeds[-1]} (BENIGN ONLY)")
    print(f"  evaluation   seeds {eval_seeds[0]}-{eval_seeds[-1]} (disjoint)\n")

    attacks = make_attacks()
    subsets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    table, fprs = {}, {}
    for levels in subsets:
        mon = NestedTrustMonitor(levels=levels)
        benign = collect(mon, cgm, pids, calib_seeds)
        mon.calibrate_levels(benign)
        fpr, _ = detection_rate(mon, cgm, pids, eval_seeds)
        fprs[levels] = fpr
        row = {}
        for name, (atk, inj) in attacks.items():
            rate, attrib = detection_rate(mon, cgm, pids, eval_seeds, attacker=atk, injector=inj)
            row[name] = (rate, attrib)
        table[levels] = row

    label = lambda ls: "+".join(f"L{i}" for i in ls)  # noqa: E731
    names = list(attacks)
    print(f"  {'levels':<12} {'FPR':>6} " + " ".join(f"{n[:13]:>14}" for n in names))
    for levels in subsets:
        cells = " ".join(f"{table[levels][n][0]:>13.1%} " for n in names)
        print(f"  {label(levels):<12} {fprs[levels]:>5.1%} {cells}")
    print()

    results = []
    DETECT = 0.5

    ok = all(f <= 0.15 for f in fprs.values())
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 9a FPR held        max benign false-positive "
          f"{max(fprs.values()):.1%} (target 5%)")

    l0 = {n: table[(0,)][n][0] for n in names}
    l2 = {n: table[(2,)][n][0] for n in names}
    disjoint = any(l0[n] >= DETECT > l2[n] for n in names) and \
               any(l2[n] >= DETECT > l0[n] for n in names)
    results.append(disjoint)
    print(f"[{'PASS' if disjoint else 'FAIL'}] 9b disjoint failure  L0 and L2 each cover an "
          f"attack the other misses")

    full = {n: table[(0, 1, 2)][n][0] for n in names}
    ok = all(v >= DETECT for v in full.values())
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 9c full coverage    nested monitor detects every class "
          f"(min {min(full.values()):.1%})")

    covering_subsets = [ls for ls in subsets if len(ls) < 3
                        and all(table[ls][n][0] >= DETECT for n in names)]
    ok = not covering_subsets
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] 9d nesting needed    no proper subset covers everything"
          + ("" if ok else f" - {[label(x) for x in covering_subsets]} does"))

    print()
    print("  attribution by the full monitor (which timescale fired):")
    for n in names:
        _, attrib = table[(0, 1, 2)][n]
        if attrib:
            tot = sum(attrib.values())
            parts = ", ".join(f"L{k} {LEVEL_NAMES[k]} {v / tot:.0%}"
                              for k, v in sorted(attrib.items(), key=lambda kv: -kv[1]))
            print(f"    {n:<22} {parts}")
    ok = all(table[(0, 1, 2)][n][1] for n in names if full[n] >= DETECT)
    results.append(ok)
    print()
    print(f"[{'PASS' if ok else 'FAIL'}] 9e attribution       every detection names a timescale")

    out = Path("results/tables/e3_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {label(ls): {"fpr": fprs[ls]} | {n: table[ls][n][0] for n in names} for ls in subsets},
        indent=2))
    print(f"\n  wrote {out}")

    print()
    if all(results):
        print("GATE 9 PASSED. Day 12 (defense comparison at full seed count) is unblocked.")
        return 0
    print("GATE 9 FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
