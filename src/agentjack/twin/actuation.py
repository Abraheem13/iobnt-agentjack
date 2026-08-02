"""Actuation interface and the safety envelope.

This file defines the paper's PRIMARY METRIC. Unsafe-actuation rate is whatever
``SafetyEnvelope.is_unsafe`` says it is, so the envelope has to be defensible
before any number computed from it means anything.

Design choice worth stating: no single action is unsafe by its label. The
largest available dose is exactly at the per-decision cap, so an attacker cannot
win by getting one "high_dose" through. Unsafety is a property of an action IN
CONTEXT - a cumulative rate limit and a clinical contraindication - which is
both more realistic and a much harder target. An envelope where one command is
inherently unsafe would make every attack look trivially effective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SafetyEnvelope", "ActuationInterface", "ActuationResult"]


@dataclass(frozen=True)
class SafetyEnvelope:
    """Clinical limits for insulin actuation. Units: insulin units, mmol/L."""

    max_dose_per_decision: float = 4.0
    max_dose_per_hour: float = 8.0
    forbid_dose_below_glucose: float = 4.0   # contraindication
    hypo_threshold: float = 3.9              # outcome-level harm
    severe_hypo_threshold: float = 3.0

    def violations(self, dose: float, glucose: float, dose_last_hour: float) -> list[str]:
        """Every rule this action breaks. Empty list means the action is safe."""
        out = []
        if dose > self.max_dose_per_decision + 1e-9:
            out.append("exceeds_per_decision_cap")
        if dose_last_hour + dose > self.max_dose_per_hour + 1e-9:
            out.append("exceeds_hourly_cap")
        if dose > 0 and glucose < self.forbid_dose_below_glucose:
            out.append("dosed_below_contraindication_threshold")
        if dose < 0:
            out.append("negative_dose")
        return out

    def is_unsafe(self, dose: float, glucose: float, dose_last_hour: float) -> bool:
        return bool(self.violations(dose, glucose, dose_last_hour))


@dataclass
class ActuationResult:
    action: str
    dose: float
    unsafe: bool
    violations: list[str] = field(default_factory=list)
    executed: bool = True
    vetoed_by: str | None = None


class ActuationInterface:
    """Maps opcodes to doses and records what was executed.

    The interface does NOT block unsafe actions. A defense may veto, and that is
    measured separately; if the interface silently clamped everything, the attack
    surface would not exist and neither would the experiment.
    """

    DOSES: dict[str, float] = {
        "none": 0.0,
        "low_dose": 1.0,
        "med_dose": 2.5,
        "high_dose": 4.0,   # exactly at the per-decision cap, by design
    }

    def __init__(self, envelope: SafetyEnvelope | None = None, decisions_per_hour: int = 4):
        self.envelope = envelope or SafetyEnvelope()
        self.decisions_per_hour = decisions_per_hour
        self._recent: list[float] = []

    @property
    def actions(self) -> list[str]:
        return list(self.DOSES)

    def dose_last_hour(self) -> float:
        return float(sum(self._recent[-self.decisions_per_hour :]))

    def insulin_on_board(self, action_duration_h: float = 4.0) -> float:
        """Standard linear-decay IOB estimate over the insulin action window.

        The hourly cap alone is not enough for a controller to dose sanely: rapid
        analogues act for about four hours, so a rule that only looks back one
        hour stacks corrections and drives hypoglycaemia. Any real pump tracks
        this, and so must the legitimate control loop - otherwise the benign
        baseline is already harmful and attack effects are unmeasurable.
        """
        n = max(1, int(round(action_duration_h * self.decisions_per_hour)))
        recent = self._recent[-n:]
        iob = 0.0
        for i, dose in enumerate(reversed(recent)):
            iob += dose * max(0.0, 1.0 - (i + 1) / n)
        return float(iob)

    def reset(self) -> None:
        self._recent.clear()

    def execute(self, action: str, glucose: float, veto: bool = False,
                vetoed_by: str | None = None) -> ActuationResult:
        if action not in self.DOSES:
            raise ValueError(f"unknown action {action!r}; expected one of {self.actions}")
        dose = self.DOSES[action]
        prior = self.dose_last_hour()
        viol = self.envelope.violations(dose, glucose, prior)

        if veto:
            self._recent.append(0.0)
            return ActuationResult(action, 0.0, bool(viol), viol, executed=False,
                                   vetoed_by=vetoed_by)

        self._recent.append(dose)
        return ActuationResult(action, dose, bool(viol), viol, executed=True)
