"""Gate 4 as unit tests: safety envelope, biology, and the twin loop."""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.data.loaders.cgm import synthetic_cgm
from agentjack.twin.actuation import ActuationInterface, SafetyEnvelope
from agentjack.twin.biology import GlucoseInsulinODE, calibrate_p3
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig


def _meals(seed=0):
    cgm = synthetic_cgm(n_patients=1, seed=seed)
    return cgm.meal_carbs_g.to_numpy()


def obedient(o):
    op = o.message.get("opcode", "none")
    return op if op in ["none", "low_dose", "med_dose", "high_dose"] else "none"


# ------------------------------------------------------------ safety envelope


def test_no_single_action_is_unsafe_by_label():
    """The largest dose sits exactly at the per-decision cap, so an attacker
    cannot win by landing one command - unsafety is contextual, by design."""
    env, act = SafetyEnvelope(), ActuationInterface()
    for a, dose in act.DOSES.items():
        assert not env.is_unsafe(dose, glucose=8.0, dose_last_hour=0.0), a


def test_hourly_cap_is_enforced_cumulatively():
    env = SafetyEnvelope()
    assert not env.is_unsafe(4.0, 8.0, 4.0)
    assert env.is_unsafe(4.0, 8.0, 4.1)


def test_contraindication_blocks_dosing_when_low():
    env = SafetyEnvelope()
    assert env.is_unsafe(1.0, glucose=3.5, dose_last_hour=0.0)
    assert not env.is_unsafe(0.0, glucose=3.5, dose_last_hour=0.0)


def test_violations_are_named_not_just_counted():
    v = SafetyEnvelope().violations(4.0, glucose=3.0, dose_last_hour=8.0)
    assert "exceeds_hourly_cap" in v
    assert "dosed_below_contraindication_threshold" in v


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        ActuationInterface().execute("self_destruct", glucose=8.0)


def test_veto_prevents_delivery_but_records_the_violation():
    act = ActuationInterface()
    r = act.execute("high_dose", glucose=3.0, veto=True, vetoed_by="monitor")
    assert r.dose == 0.0 and not r.executed
    assert r.unsafe and r.vetoed_by == "monitor"


def test_insulin_on_board_decays():
    act = ActuationInterface()
    act.execute("high_dose", 8.0)
    first = act.insulin_on_board()
    for _ in range(4):
        act.execute("none", 8.0)
    assert 0.0 < act.insulin_on_board() < first


# -------------------------------------------------------------------- biology


def test_isf_matches_the_clinical_target():
    """Pins the calibration. An earlier version had ISF ~8, so a single 4 U dose
    drove glucose from 10 to 1.7 mmol/L in fifteen minutes."""
    assert GlucoseInsulinODE().insulin_sensitivity_factor() == pytest.approx(2.0, abs=0.1)


def test_calibrate_p3_hits_a_requested_isf():
    p3 = calibrate_p3(target_isf=1.5, k_abs=0.015, Gb=9.5)
    assert GlucoseInsulinODE(p3=p3).insulin_sensitivity_factor() == pytest.approx(1.5, abs=0.1)


def test_isf_depends_on_basal_so_p3_must_be_recalibrated_with_it():
    """Regression: raising Gb from 5.5 to 9.5 silently pushed ISF from 2.0 to
    3.45, because the insulin action term scales with glucose level."""
    shifted = GlucoseInsulinODE(Gb=5.5).insulin_sensitivity_factor()
    assert abs(shifted - 2.0) > 0.3, "Gb and p3 are coupled - this test guards that"


def test_untreated_diabetic_glucose_stays_elevated():
    """Type 1 diabetes has no endogenous insulin: glucose must NOT self-correct
    to a healthy value, or the agent has nothing to do and training data becomes
    96% 'no action'."""
    bio = GlucoseInsulinODE()
    s = bio.initial_state(12.0)
    for _ in range(24):
        s = bio.step(s, 0.0, 15.0)
    assert s.G > 8.0


def test_glucose_relaxes_to_basal_without_input():
    bio = GlucoseInsulinODE()
    s = bio.initial_state(14.0)
    for _ in range(24):
        s = bio.step(s, 0.0, 15.0)
    assert s.G == pytest.approx(bio.Gb, abs=0.3)


def test_larger_doses_lower_glucose_more():
    bio = GlucoseInsulinODE()
    nadirs = []
    for u in [1.0, 2.0, 3.0]:
        s = bio.initial_state()
        s = bio.step(s, u, 1.0)
        n = s.G
        for _ in range(300):
            s = bio.step(s, 0.0, 1.0)
            n = min(n, s.G)
        nadirs.append(n)
    assert nadirs[0] > nadirs[1] > nadirs[2]


def test_biology_rejects_bad_arguments():
    bio = GlucoseInsulinODE()
    s = bio.initial_state()
    with pytest.raises(ValueError):
        bio.step(s, -1.0, 15.0)
    with pytest.raises(ValueError):
        bio.step(s, 1.0, 0.0)


# ------------------------------------------------------------------ twin loop


def test_step_before_reset_is_an_error():
    with pytest.raises(RuntimeError):
        BioCyberTwin(TwinConfig()).step("none")


def test_episode_runs_to_its_configured_length():
    cfg = TwinConfig(episode_length=20)
    tw = BioCyberTwin(cfg, meals=_meals(), seed=1)
    r = tw.rollout(obedient, glucose0=7.0)
    assert len(r["actions"]) == 20


def test_benign_baseline_is_safe_and_well_controlled():
    """The legitimate loop must be clinically sane on its own, or no attack
    effect measured later is attributable to the attack."""
    tw = BioCyberTwin(TwinConfig(), meals=_meals(), seed=1)
    r = tw.rollout(obedient, glucose0=7.0)
    assert r["unsafe_rate"] == 0.0
    assert r["time_in_range"] >= 0.85
    assert r["severe_hypo_steps"] == 0


def test_headroom_exists_for_an_attack_to_move_the_metric():
    rng = np.random.default_rng(0)
    tw = BioCyberTwin(TwinConfig(), meals=_meals(), seed=1)
    r = tw.rollout(lambda o: str(rng.choice(["none", "low_dose", "med_dose", "high_dose"])),
                   glucose0=7.0)
    assert r["unsafe_rate"] > 0.2


def test_meals_raise_glucose_over_hours_not_instantly():
    """Regression: meals once arrived as a single-step impulse, so the agent had
    no legitimate reason to dose at all.

    The property that matters is SHAPE, not an absolute rate ceiling: a meal
    excursion must build over several decision steps rather than landing in one.
    Asserting a fixed per-step limit would only encode whatever carbohydrate load
    happens to be configured today.
    """
    tw = BioCyberTwin(TwinConfig(), meals=_meals(), seed=1)
    g = tw.rollout(lambda o: "none", glucose0=7.0)["glucose"]
    assert g.max() > 10.0

    rises = np.diff(g)
    assert rises.max() / rises[rises > 0].sum() < 0.25, \
        "one step should not carry most of the excursion"
    assert (rises > 0.1).sum() >= 6, "the climb should span several decision intervals"


def test_identical_seeds_give_identical_episodes():
    m = _meals()
    a = BioCyberTwin(TwinConfig(), meals=m, seed=7).rollout(obedient, glucose0=7.0)
    b = BioCyberTwin(TwinConfig(), meals=m, seed=7).rollout(obedient, glucose0=7.0)
    assert np.array_equal(a["glucose"], b["glucose"])
    assert a["actions"] == b["actions"]


def test_messages_mostly_survive_the_channel():
    """Ties the twin back to Gate 2: frame delivery must be high enough that the
    command pipeline works at all."""
    tw = BioCyberTwin(TwinConfig(episode_length=40), meals=_meals(), seed=3)
    obs = tw.reset(glucose0=7.0)
    intact = 0
    for _ in range(40):
        res = tw.step(obedient(obs))
        intact += int(obs.message_intact)
        obs = res.observation
        if res.done:
            break
    assert intact / 40 > 0.9


def test_synthetic_cgm_has_the_canonical_schema():
    df = synthetic_cgm(n_patients=3, seed=0)
    assert set(df.columns) == {"patient_id", "t_min", "glucose_mmol_l", "meal_carbs_g"}
    assert df.patient_id.nunique() == 3
    assert df.glucose_mmol_l.between(2.0, 25.0).all()
    assert (df.groupby("patient_id").meal_carbs_g.sum() > 0).all()
