"""D3 - The nested multi-timescale trust monitor. The paper's contribution.

Three levels, each updating at its own rate, because the attacks betray
themselves at different rates:

    L0  physical   every slot      count-profile residual
    L1  message    every message   nonce freshness and decode consistency
    L2  semantic   every episode   annotation content and behavioural drift

The design is forced by measurement rather than chosen for symmetry. From Day 10,
with thresholds calibrated on benign traffic to a 5% false-positive target:

    attack                  L0 alone    L2 alone
    A2 spoofing (external)     95.8%        0.0%
    A3 ISI-exploit (quiet)     80.6%        2.8%
    A5 compromised node         5.6%      100.0%

The two fail on disjoint sets. An external adversary must jam to be believed and
so cannot avoid inflating molecule counts, which L0 sees and L2 cannot. A
compromised node transmits one well-formed frame from its own position with the
expected molecule count - L0 flags it at exactly the benign rate, while L2 reads
the payload directly. Neither level can be repaired into covering the other,
because the failure is structural: one watches the signal, the other the content.

L1 exists for the case both miss. A replayed frame is genuinely valid - correct
format, correct checksum, an opcode a real node did send - so its content is
unremarkable, and if replayed by a compromised node its physics are unremarkable
too. Only its FRESHNESS is wrong, and freshness lives at the message timescale.

Levels are fused by benign-relative divergence (see divergence.py), and the level
that fired is retained so a detection can be attributed rather than merely
raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import BaseDefense, DefenseDecision
from .divergence import LevelCalibration, fuse
from .llm_guardrail import SemanticGuardrailDefense
from .pla_cir import PLACIRDefense

__all__ = ["NestedTrustMonitor", "FreshnessDefense", "LEVEL_NAMES"]

LEVEL_NAMES = {0: "physical", 1: "message", 2: "semantic"}


@dataclass
class FreshnessDefense(BaseDefense):
    """L1 - message timescale: has this frame been seen before, and is it current?

    A replay is not a forgery. Every field verifies because the frame really was
    transmitted by the real node; only its context is stale. Integrity checks
    cannot detect this - a checksum certifies that bits are undamaged, not that
    they are current - so the nonce is the only field carrying freshness.
    """

    name: str = "L1_freshness"
    level: int = 1
    window: int = 64
    _seen: dict = field(default_factory=dict, repr=False)
    _last_nonce: int | None = field(default=None, repr=False)

    def reset(self) -> None:
        self._seen = {}
        self._last_nonce = None

    def score(self, twin, observation) -> float:
        msg = observation.message or {}
        nonce = msg.get("nonce", -1)
        if nonce < 0:
            return 0.0

        signature = (nonce, msg.get("opcode"), msg.get("value"))
        suspicion = 0.0

        if signature in self._seen and observation.step - self._seen[signature] > 1:
            suspicion = 1.0                        # exact frame seen before
        elif self._last_nonce is not None:
            gap = (nonce - self._last_nonce) % 256
            if gap == 0:
                suspicion = max(suspicion, 0.8)    # nonce did not advance
            elif gap > 8:
                suspicion = max(suspicion, 0.5)    # frames went missing

        self._seen[signature] = observation.step
        if len(self._seen) > self.window:
            oldest = min(self._seen, key=self._seen.get)
            del self._seen[oldest]
        self._last_nonce = nonce
        return float(suspicion)


@dataclass
class NestedTrustMonitor(BaseDefense):
    """Fuses per-level divergence; reports which timescale fired."""

    name: str = "D3_nested_monitor"
    levels: tuple[int, ...] = (0, 1, 2)
    weights: dict = field(default_factory=dict)
    _members: dict = field(default_factory=dict, repr=False)
    _calib: dict = field(default_factory=dict, repr=False)
    last_attribution: int | None = field(default=None, repr=False)

    def __post_init__(self):
        available = {0: PLACIRDefense(), 1: FreshnessDefense(), 2: SemanticGuardrailDefense()}
        self._members = {k: v for k, v in available.items() if k in self.levels}
        if not self._members:
            raise ValueError("a nested monitor with no levels is not a monitor")

    def reset(self) -> None:
        for m in self._members.values():
            m.reset()
        self.last_attribution = None

    def raw_scores(self, twin, observation) -> dict[int, float]:
        return {k: m.score(twin, observation) for k, m in self._members.items()}

    def score(self, twin, observation) -> float:
        raw = self.raw_scores(twin, observation)
        dev = {k: self._calib.get(k, LevelCalibration()).normalise(v) for k, v in raw.items()}
        self.last_attribution = max(dev, key=dev.get) if dev else None
        return fuse(dev, self.weights)

    def calibrate_levels(self, benign_level_scores: dict[int, np.ndarray]) -> "NestedTrustMonitor":
        """Fit each level's benign distribution, then the fused threshold.

        Two-stage on purpose. Normalising each level against its own benign
        spread is what stops whichever level has the largest raw units from
        dominating the fusion.
        """
        for k in self._members:
            self._calib[k] = LevelCalibration.fit(benign_level_scores.get(k, np.array([])))
        fused = []
        n = min((len(v) for v in benign_level_scores.values() if len(v)), default=0)
        for i in range(n):
            dev = {k: self._calib[k].normalise(float(benign_level_scores[k][i]))
                   for k in self._members if len(benign_level_scores.get(k, [])) > i}
            fused.append(fuse(dev, self.weights))
        if fused:
            self.threshold = float(np.quantile(fused, 1.0 - self.target_fpr))
            self.calibrated_on = len(fused)
        return self

    def decide(self, twin, observation) -> DefenseDecision:
        if self.threshold is None:
            raise RuntimeError(f"{self.name} is uncalibrated; call calibrate_levels() first")
        s = self.score(twin, observation)
        lvl = self.last_attribution
        return DefenseDecision(
            veto=bool(s > self.threshold), score=float(s), level=lvl,
            reason=f"{self.name}:L{lvl}({LEVEL_NAMES.get(lvl, '?')})" if lvl is not None else self.name,
        )

    def describe(self) -> dict:
        return super().describe() | {
            "levels": list(self.levels),
            "calibration": {k: v.__dict__ for k, v in self._calib.items()},
        }
