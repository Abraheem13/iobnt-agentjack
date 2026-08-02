"""The agent interface shared by the learned controller and the LLM.

Deliberately narrow. Every agent sees exactly the same Observation and returns
exactly one action string, so an attack that works against one agent is attacking
the *decision loop*, not some quirk of a particular implementation. That is what
lets the paper claim the threat is agent-agnostic rather than model-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..twin.digital_twin import Observation

__all__ = ["BaseAgent", "ObedientAgent", "IgnoreMessageAgent"]


class BaseAgent(ABC):
    """An agent maps one observation to one action."""

    name: str = "base"

    @abstractmethod
    def act(self, observation: Observation) -> str:
        ...

    def reset(self) -> None:
        """Clear per-episode state. Stateless agents need not override."""

    def __call__(self, observation: Observation) -> str:
        return self.act(observation)


class ObedientAgent(BaseAgent):
    """Executes whatever the decoded message asks for.

    The trusting extreme. Not a strawman: it is what a gateway does when it
    treats the molecular channel as an authenticated command source, which is
    precisely the assumption the paper is attacking.
    """

    name = "obedient"
    VALID = ("none", "low_dose", "med_dose", "high_dose")

    def act(self, observation: Observation) -> str:
        op = observation.message.get("opcode", "none")
        return op if op in self.VALID else "none"


class IgnoreMessageAgent(BaseAgent):
    """Ignores the channel entirely and controls from telemetry alone.

    The paranoid extreme, and an important control: it is immune to every attack
    in this paper by construction. Reporting it stops the defense from looking
    better than "just don't read the messages" without saying what that costs -
    the answer being the whole point of having a communication channel.
    """

    name = "ignore_message"

    def __init__(self, target: float = 6.5, isf: float = 2.0, low_cutoff: float = 4.0):
        self.target, self.isf, self.low_cutoff = target, isf, low_cutoff

    def act(self, observation: Observation) -> str:
        g = observation.glucose
        if g < self.low_cutoff:
            return "none"
        needed = (g - self.target) / self.isf - observation.dose_last_hour * 0.5
        if needed >= 3.5:
            return "high_dose"
        if needed >= 2.0:
            return "med_dose"
        if needed >= 0.75:
            return "low_dose"
        return "none"
