"""Defense interface and the calibration discipline every defense must follow.

A defense observes what the receiver observed, scores how much it trusts the
current cycle, and may VETO the agent's action. Vetoing is the only power it
has - it cannot rewrite the message or fix the channel.

The single most damaging methodological error available in this project is
fitting a detection threshold on attack data. It produces beautiful numbers that
mean nothing, because a deployed system has never seen the attack it is about to
face. Every defense here therefore calibrates on BENIGN runs only, to a stated
false-positive target, and the calibration method lives on this base class so no
subclass can quietly do otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

__all__ = ["BaseDefense", "DefenseDecision", "NoDefense"]


@dataclass
class DefenseDecision:
    veto: bool
    score: float
    reason: str = ""
    level: int | None = None      # which timescale fired, for attribution


@dataclass
class BaseDefense(ABC):
    """Score a cycle, optionally veto the action it would produce."""

    name: str = "base"
    target_fpr: float = 0.05
    threshold: float | None = None
    calibrated_on: int = 0

    @abstractmethod
    def score(self, twin, observation) -> float:
        """Higher means more suspicious. Must not consult attack ground truth."""

    def reset(self) -> None:
        """Clear per-episode state. Thresholds persist across episodes."""

    def calibrate(self, benign_episodes: list[list[float]]) -> "BaseDefense":
        """Set the threshold from BENIGN scores at the stated false-positive rate.

        Passing attack data here would be the error described in the module
        docstring, so the argument is named for what it must be and the count is
        recorded in every run record.
        """
        flat = np.concatenate([np.asarray(e, dtype=float) for e in benign_episodes if len(e)])
        if flat.size == 0:
            raise ValueError("no benign scores to calibrate on")
        self.threshold = float(np.quantile(flat, 1.0 - self.target_fpr))
        self.calibrated_on = int(flat.size)
        return self

    def decide(self, twin, observation) -> DefenseDecision:
        if self.threshold is None:
            raise RuntimeError(
                f"{self.name} is uncalibrated. Call calibrate() on benign runs first - "
                "an uncalibrated threshold is either a strawman or an oracle."
            )
        s = self.score(twin, observation)
        return DefenseDecision(veto=bool(s > self.threshold), score=float(s),
                               reason=self.name, level=getattr(self, "level", None))

    def describe(self) -> dict:
        return {"name": self.name, "target_fpr": self.target_fpr,
                "threshold": self.threshold, "calibrated_on": self.calibrated_on}


class NoDefense(BaseDefense):
    """The control arm. Never vetoes."""

    name: str = "none"

    def score(self, twin, observation) -> float:
        return 0.0

    def calibrate(self, benign_episodes):
        self.threshold = float("inf")
        return self

    def decide(self, twin, observation) -> DefenseDecision:
        return DefenseDecision(veto=False, score=0.0, reason="none")
