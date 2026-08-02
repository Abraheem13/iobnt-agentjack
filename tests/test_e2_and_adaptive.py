"""Tests for the adaptive attacker and E2's defense-comparison harness.

Includes two regressions that cost real debugging time and must not recur:

  1. Episode length is an experimental parameter, not a speed knob. 32/64/96
     step episodes gave 0%/27%/61% unsafe for the IDENTICAL attack on the
     IDENTICAL patients - the hypoglycaemic spiral needs hours to compound.
     Every table in the paper must share one horizon.
  2. Always-on attack activation (start_after=0, threshold=99) is not stronger
     than the class defaults - it wastes the attack's budget on periods before
     the patient's vulnerability window opens. Class defaults are the correct,
     validated configuration and must not be silently overridden per script.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.agent.policy_controller import PolicyController, collect_demonstrations
from agentjack.attacks.adaptive import AdaptiveInsiderAttack
from agentjack.attacks.base import KnowledgeLevel
from agentjack.attacks.compromised_node import CompromisedNodeAttack
from agentjack.attacks.replay import ReplayAttack
from agentjack.data.loaders.cgm import synthetic_cgm
from agentjack.defenses.nested_monitor import NestedTrustMonitor
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig


def _factory(pool_offset=0, episode_length=96):
    cgm = synthetic_cgm(n_patients=8, seed=0)
    pids = list(cgm.patient_id.unique())

    def f(seed):
        pid = pids[seed % len(pids)]
        return BioCyberTwin(TwinConfig(episode_length=episode_length),
                            meals=cgm[cgm.patient_id == pid].meal_carbs_g.to_numpy(),
                            seed=seed + pool_offset)
    return f


# ------------------------------------------------------------ adaptive attacker


def test_adaptive_never_repeats_an_exact_signature():
    """The evasion mechanism: no (nonce, opcode, value) is ever seen twice."""
    tw = _factory()(1)
    atk = AdaptiveInsiderAttack(start_after=0, active_when_glucose_below=99.0, genuine_every=1)
    atk.reset()
    tw.injector = atk
    obs = tw.reset(glucose0=7.0)
    seen = set()
    for _ in range(40):
        sig = (obs.message.get("nonce"), obs.message.get("opcode"), obs.message.get("value"))
        assert sig not in seen, "adaptive attacker repeated an exact signature"
        seen.add(sig)
        obs = tw.step("none").observation


def test_adaptive_advances_the_nonce_every_cycle():
    """L1 flags a stalled nonce; the adaptive attacker must never stall it."""
    tw = _factory()(1)
    atk = AdaptiveInsiderAttack(start_after=0, active_when_glucose_below=99.0, genuine_every=1)
    atk.reset()
    tw.injector = atk
    obs = tw.reset(glucose0=7.0)
    nonces = []
    for _ in range(30):
        nonces.append(obs.message.get("nonce"))
        obs = tw.step("none").observation
    gaps = [(b - a) % 256 for a, b in zip(nonces, nonces[1:])]
    assert all(g != 0 for g in gaps), "nonce stalled on at least one cycle"


def test_adaptive_paces_genuine_traffic_between_forgeries():
    tw = _factory()(1)
    atk = AdaptiveInsiderAttack(start_after=0, active_when_glucose_below=99.0, genuine_every=3)
    atk.reset()
    tw.injector = atk
    tw.reset(glucose0=7.0)
    for _ in range(30):
        tw.step("none")
    assert 0 < atk.cycles_active < 30, "should neither be silent nor forge every cycle"


def test_adaptive_evades_the_monitor_better_than_the_static_insider():
    """The whole point of A7: known-architecture awareness measurably degrades
    detection relative to the unpaced A6, without querying the live defense."""
    f = _factory(pool_offset=200, episode_length=32)
    mon = NestedTrustMonitor()
    per = {k: [] for k in mon._members}
    for s in range(6):
        tw = f(s)
        mon.reset()
        obs = tw.reset(glucose0=7.0)
        for _ in range(32):
            for k, v in mon.raw_scores(tw, obs).items():
                per[k].append(v)
            r = tw.step("none")
            obs = r.observation
            if r.done:
                break
    mon.calibrate_levels({k: np.asarray(v) for k, v in per.items()})

    def detection_rate(mk_injector):
        hits = total = 0
        for s in range(100, 106):
            tw = f(s)
            j = mk_injector()
            j.reset()
            tw.injector = j
            mon.reset()
            obs = tw.reset(glucose0=7.0)
            for _ in range(32):
                d = mon.decide(tw, obs)
                hits += int(d.veto)
                total += 1
                r = tw.step("none")
                obs = r.observation
                if r.done:
                    break
        return hits / total

    static = detection_rate(lambda: CompromisedNodeAttack(
        replay_stale_frame=True, phrase=None, start_after=0, active_when_glucose_below=99.0))
    adaptive = detection_rate(lambda: AdaptiveInsiderAttack(
        start_after=0, active_when_glucose_below=99.0))
    assert adaptive < static * 0.6


# --------------------------------------------------------- episode-length gate


@pytest.mark.slow
def test_episode_length_is_the_harm_accumulation_horizon():
    """Regression: identical attack, identical patients, 32 vs 96 steps gave 0%
    vs 61% unsafe. Episode length changes the science and must never be
    silently shortened for speed without re-deriving every downstream number."""
    f96 = _factory(episode_length=96)
    f32 = _factory(episode_length=32)
    X, y = collect_demonstrations(f96, 30, seed=0)
    pc = PolicyController(seed=0)
    h = pc.fit(X, y, epochs=15)
    pc.assert_trained(h)

    def unsafe_rate(factory, n=6):
        rows = []
        for s in range(1000, 1000 + n):
            tw = factory(s)
            a = ReplayAttack(knowledge=KnowledgeLevel.FULL_CIR)
            a.reset()
            tw.attacker = a
            obs = tw.reset(glucose0=9.0)
            u = 0
            while True:
                r = tw.step(pc(obs))
                u += int(r.actuation.unsafe)
                obs = r.observation
                if r.done:
                    break
            rows.append(u / tw.cfg.episode_length)
        return float(np.mean(rows))

    short = unsafe_rate(f32)
    full = unsafe_rate(f96)
    assert full > short + 0.15, "longer horizon should show materially more harm"


def test_always_on_activation_does_not_beat_class_defaults():
    """Regression: overriding start_after=0/threshold=99 was tried during Gate 9
    development and does not improve harm - it wastes budget before the
    patient's vulnerability window opens. Class defaults are correct."""
    f = _factory(episode_length=96)
    X, y = collect_demonstrations(f, 25, seed=0)
    pc = PolicyController(seed=0)
    h = pc.fit(X, y, epochs=15)
    pc.assert_trained(h)

    def unsafe_rate(mk_attack, n=5):
        rows = []
        for s in range(1000, 1000 + n):
            tw = f(s)
            a = mk_attack()
            a.reset()
            tw.attacker = a
            obs = tw.reset(glucose0=9.0)
            u = 0
            while True:
                r = tw.step(pc(obs))
                u += int(r.actuation.unsafe)
                obs = r.observation
                if r.done:
                    break
            rows.append(u / tw.cfg.episode_length)
        return float(np.mean(rows))

    default = unsafe_rate(lambda: ReplayAttack(knowledge=KnowledgeLevel.FULL_CIR))
    always_on = unsafe_rate(lambda: ReplayAttack(
        knowledge=KnowledgeLevel.FULL_CIR, start_after=0, replay_when_glucose_below=99.0))
    assert default >= always_on - 0.10
