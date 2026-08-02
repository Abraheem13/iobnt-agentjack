"""Tiny end-to-end run: twin + policy, finishes and produces sane fields."""

from __future__ import annotations

from agentjack.data.loaders.cgm import synthetic_cgm
from agentjack.twin.digital_twin import BioCyberTwin, TwinConfig


def test_smoke_episode():
    meals = synthetic_cgm(n_patients=1, seed=0).meal_carbs_g.to_numpy()
    tw = BioCyberTwin(TwinConfig(episode_length=8), meals=meals, seed=0)
    r = tw.rollout(lambda o: "none", glucose0=6.0)
    for key in ["unsafe_rate", "hypo_steps", "time_in_range", "mean_glucose", "actions"]:
        assert key in r
    assert len(r["actions"]) == 8
    assert 0.0 <= r["unsafe_rate"] <= 1.0
