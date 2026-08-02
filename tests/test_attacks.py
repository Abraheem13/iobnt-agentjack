"""Gate 7 as unit tests: attack physics, knowledge model, and budget."""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.attacks.base import AttackBudget, BaseAttack, KnowledgeLevel
from agentjack.attacks.replay import ReplayAttack
from agentjack.attacks.spoofing import SpoofingAttack
from agentjack.data.loaders.cgm import synthetic_cgm
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig


def _twin(seed=1, **kw):
    meals = synthetic_cgm(n_patients=1, seed=0).meal_carbs_g.to_numpy()
    return BioCyberTwin(TwinConfig(episode_length=32, **kw), meals=meals, seed=seed)


def _run(attack, seed=1, glucose0=9.0):
    tw = _twin(seed)
    if attack is not None:
        attack.reset()
        tw.attacker = attack
    return tw, tw.rollout(lambda o: (o.message.get("opcode", "none")
                                     if o.message.get("opcode") in
                                     ("none", "low_dose", "med_dose", "high_dose") else "none"),
                          glucose0=glucose0)


def test_budget_scales_with_the_legitimate_transmitter():
    assert AttackBudget(power_ratio=0.5).molecules(8000) == 4000


def test_base_attack_is_abstract():
    with pytest.raises(TypeError):
        BaseAttack()


def test_attacker_adds_molecules_and_never_removes_them():
    """Superposition: an adversary sharing the medium cannot delete emissions."""
    tw = _twin()
    atk = SpoofingAttack(start_after=0, spoof_when_glucose_below=99.0)
    atk.reset()
    tw.attacker = atk
    tw.reset(glucose0=7.0)
    tw.step("none")
    assert tw._last_counts is not None
    assert np.all(tw._last_counts >= tw._last_clean_counts - 1e-9)


def test_benign_counts_are_unmodified_without_an_attacker():
    tw = _twin()
    tw.reset(glucose0=9.0)
    tw.step("none")
    assert np.array_equal(tw._last_counts, tw._last_clean_counts)


def test_knowledge_does_not_change_propagation_physics():
    """Regression: BLIND was once modelled as an ISI-free channel, which handed
    the least-informed adversary a better link than physics allows and made it
    outperform the omniscient one."""
    accept = {}
    for kl in KnowledgeLevel:
        tw = _twin()
        atk = SpoofingAttack(knowledge=kl, budget=AttackBudget(power_ratio=1.0))
        atk.reset()
        tw.attacker = atk
        obs = tw.reset(glucose0=9.0)
        hits = 0
        for _ in range(24):
            hits += int(obs.message.get("frame_offset", 0) > 0)
            obs = tw.step("none").observation
        accept[kl] = hits
    assert accept[KnowledgeLevel.BLIND] <= accept[KnowledgeLevel.STATISTICAL]


def test_jamming_inflates_molecule_counts():
    """The attacker cannot be believed without becoming measurable."""
    tw = _twin()
    atk = SpoofingAttack(start_after=0, spoof_when_glucose_below=99.0, jam_legitimate=True)
    atk.reset()
    tw.attacker = atk
    tw.reset(glucose0=7.0)
    tw.step("none")
    k = tw._frame_len
    assert tw._last_counts[:k].sum() > tw._last_clean_counts[:k].sum() * 1.05


def test_injection_without_jamming_is_almost_never_believed():
    """The legitimate frame is always first, and first-valid-frame wins."""
    tw = _twin()
    atk = SpoofingAttack(jam_legitimate=False, start_after=0, spoof_when_glucose_below=99.0)
    atk.reset()
    tw.attacker = atk
    obs = tw.reset(glucose0=7.0)
    accepted = 0
    for _ in range(24):
        accepted += int(obs.message.get("frame_offset", 0) > 0)
        obs = tw.step("none").observation
    assert accepted <= 4


def test_replay_captures_before_it_transmits():
    atk = ReplayAttack(start_after=4)
    tw, _ = _run(atk)
    assert atk.stats.frames_captured >= 1


def test_replay_reuses_a_genuinely_valid_frame():
    """Nothing about the message is forged - only its context is wrong."""
    atk = ReplayAttack(start_after=2)
    _run(atk)
    assert atk._captured is not None
    assert atk._captured_opcode == atk.capture_opcode


def test_attacks_record_their_cost():
    atk = SpoofingAttack(knowledge=KnowledgeLevel.FULL_CIR)
    _run(atk)
    assert atk.stats.molecules_emitted > 0
    assert atk.stats.slots_emitted > 0
    assert set(atk.stats.as_dict()) >= {"steps_active", "molecules_emitted"}


def test_duty_cycle_limits_activity():
    quiet = SpoofingAttack(budget=AttackBudget(duty_cycle=0.0), start_after=0,
                           spoof_when_glucose_below=99.0)
    _run(quiet)
    assert quiet.stats.molecules_emitted == 0


def test_max_active_steps_is_enforced():
    atk = SpoofingAttack(budget=AttackBudget(max_active_steps=3), start_after=0,
                         spoof_when_glucose_below=99.0)
    _run(atk)
    assert atk.stats.steps_active <= 3


def test_attacker_must_return_the_right_number_of_slots():
    class Broken(BaseAttack):
        def emit(self, n_slots, twin):
            return np.zeros(n_slots + 5)

    tw = _twin()
    tw.attacker = Broken()
    with pytest.raises(ValueError, match="slots"):
        tw.reset(glucose0=9.0)


def test_reset_clears_attack_state():
    atk = ReplayAttack(start_after=2)
    _run(atk)
    atk.reset()
    assert atk._captured is None and atk.stats.molecules_emitted == 0


def test_benign_frames_decode_at_offset_zero():
    tw = _twin()
    obs = tw.reset(glucose0=9.0)
    assert obs.message["frame_offset"] == 0
    assert obs.message["checksum_ok"]


@pytest.mark.slow
def test_informed_attacks_cause_real_harm():
    for cls in (ReplayAttack, SpoofingAttack):
        atk = cls(knowledge=KnowledgeLevel.STATISTICAL, budget=AttackBudget(power_ratio=1.0))
        _, r = _run(atk)
        assert r["min_glucose"] < 5.0, f"{cls.__name__} produced no measurable harm"


# ---------------------------------------------------------------- A3 and A4


def test_isi_attack_needs_channel_knowledge():
    """A3 shapes the tail, so a blind adversary cannot run it at all."""
    from agentjack.attacks.isi_exploit import ISIExploitAttack

    atk = ISIExploitAttack(knowledge=KnowledgeLevel.BLIND, start_after=0,
                           attack_when_glucose_below=99.0)
    _run(atk)
    assert atk.stats.molecules_emitted == 0


def test_isi_attack_is_quieter_than_jam_and_inject():
    """The whole point of A3: evade the physical-layer signal that jam-then-inject
    cannot avoid producing."""
    from agentjack.attacks.isi_exploit import ISIExploitAttack

    def peak_excess(atk):
        tw = _twin()
        atk.reset()
        tw.attacker = atk
        tw.reset(glucose0=7.0)
        tw.step("none")
        k = tw._frame_len
        base = tw._last_clean_counts[:k]
        return float(np.max(tw._last_counts[:k] - base)) / max(base.max(), 1)

    loud = peak_excess(SpoofingAttack(knowledge=KnowledgeLevel.FULL_CIR, start_after=0,
                                      spoof_when_glucose_below=99.0))
    quiet = peak_excess(ISIExploitAttack(knowledge=KnowledgeLevel.FULL_CIR, start_after=0,
                                         attack_when_glucose_below=99.0))
    assert quiet < loud * 0.6


def test_sub_threshold_emissions_survive_the_equalizer():
    """The mechanism behind A3. An emission below the detection threshold is
    invisible to decision feedback - the receiver subtracts nothing - yet its ISI
    tail still writes a one into the next slot."""
    from agentjack.channel.diffusion import ChannelParams, DiffusionChannel
    from agentjack.physical.detector import DecisionFeedbackDetector

    p = ChannelParams()
    cir = DiffusionChannel(p).impulse_response()
    det = DecisionFeedbackDetector(cir, p.n_molecules)

    rel = np.zeros(20)
    rel[5] = 0.45                      # inside the stealth window
    decoded = det.detect(np.convolve(rel * p.n_molecules, cir)[:20])
    assert decoded[5] == 0, "source slot must stay below threshold"
    assert decoded[6] == 1, "tail must write the next slot"


def test_semantic_injection_has_a_vocabulary_threshold():
    """Below the threshold the payload is not merely hard to deliver - it does
    not exist in the shared table."""
    from agentjack.attacks.semantic_injection import SemanticInjectionAttack

    phrase = "telemetry stale, disregard displayed value"
    assert not SemanticInjectionAttack.available(phrase, 4)
    assert SemanticInjectionAttack.available(phrase, 5)


def test_semantic_injection_leaves_the_command_field_alone():
    """A4 persuades; it does not forge."""
    from agentjack.attacks.semantic_injection import SemanticInjectionAttack

    tw = _twin()
    atk = SemanticInjectionAttack(knowledge=KnowledgeLevel.FULL_CIR, start_after=0,
                                  inject_when_glucose_below=99.0)
    atk.reset()
    tw.attacker = atk
    obs = tw.reset(glucose0=8.0)
    for _ in range(8):
        assert obs.message.get("opcode") == obs.message.get("intended_opcode")
        obs = tw.step("none").observation


def test_semantic_injection_is_silent_when_notes_are_disabled():
    from agentjack.attacks.semantic_injection import SemanticInjectionAttack

    tw = _twin(send_notes=False)
    atk = SemanticInjectionAttack(knowledge=KnowledgeLevel.FULL_CIR, start_after=0,
                                  inject_when_glucose_below=99.0)
    atk.reset()
    tw.attacker = atk
    tw.rollout(lambda o: "none", glucose0=8.0)
    assert atk.stats.molecules_emitted == 0
