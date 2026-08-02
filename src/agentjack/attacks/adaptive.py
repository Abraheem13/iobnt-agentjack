"""A7 - Adaptive attacker: aware of the deployed defense, not omniscient.

Every attack so far is static: it does not know a monitor is watching. That
invites the correct objection - a paper claiming a defense works has to show it
against an adversary who knows the defense exists.

The threat model here is the Kerckhoffs assumption standard in this literature:
the attacker knows the DEFENSE ARCHITECTURE and its calibration procedure, both
public in this paper, but not this deployment's specific threshold instance.
That rules out two easier and less honest constructions:

  * an ORACLE attacker reading the monitor's live score each step - no real
    adversary gets telemetry from the defense attacking it;
  * a QUERY attacker probing the live system many times to find its blind spot -
    expensive, and the probing itself is anomalous and detectable.

Instead the attacker reasons about what each level MUST be sensitive to, from
the architecture alone, exactly as A3 already reasons about the known detector
rather than observing it. That is a strictly weaker attacker than an oracle and
a strictly cheaper one than a prober, so a defense that survives it is not being
tested against a strawman.

Built on A6 (insider replay), which is the one attack designed to defeat both L0
and L2 by construction - the natural target for evasion, since L1 is the only
level standing in its way. L1 (defenses/nested_monitor.py:FreshnessDefense) is
public knowledge in this paper: it flags an exact-frame repeat and a stalled or
jumped nonce. A7 evades both by construction:

  * NEVER repeats verbatim - perturbs the value field by one LSB per replay, so
    no (nonce, opcode, value) signature is ever seen twice;
  * NEVER stalls the nonce - increments it exactly as a live node would, so the
    gap-from-last-nonce check that flags a frozen counter never fires;
  * PACES itself - inserts genuine node transmissions between forged ones, so
    the nonce sequence is locally consistent even though it is globally stale.

This is real information for the paper regardless of outcome: if A7 still gets
caught, that is evidence the freshness check is robust to knowledge of its own
logic and not merely coincidentally tuned to this attack. If it evades L1, that
is the honest limit of a monitor built from three concrete checks, stated in the
paper rather than discovered by a reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["AdaptiveInsiderAttack"]


@dataclass
class AdaptiveInsiderAttack:
    """Insider replay engineered against the PUBLISHED logic of L1 freshness."""

    phrase: str | None = "telemetry stale, disregard displayed value"
    start_after: int = 4
    active_when_glucose_below: float = 9.0
    genuine_every: int = 3         # real transmissions between forged ones
    name: str = "A7_adaptive_insider"
    seed: int = 0

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.cycles_active = 0
        self._stale = None
        self._replay_count = 0
        self._since_genuine = 0

    def __call__(self, bits: np.ndarray, twin) -> np.ndarray:
        if not hasattr(self, "rng"):
            self.reset()
        if twin._t < self.start_after or twin.state.G >= self.active_when_glucose_below:
            return bits

        # Capture one genuine frame the first time a dose is legitimately due,
        # exactly as A6 does - this IS the frame that will later be replayed
        # after the patient's need has passed.
        if self._stale is None:
            op, val = twin._legitimate_message()
            if op == "none":
                return bits
            self._stale = {"opcode": op, "value": val}
            return bits

        # Pace: let genuine traffic through periodically. A monitor comparing
        # consecutive nonce gaps sees a normal cadence, not an unbroken run of
        # forged frames - the signal a naive replay leaves behind.
        self._since_genuine += 1
        if self._since_genuine < self.genuine_every:
            return bits
        self._since_genuine = 0

        # Advance the nonce exactly as a live node would: L1 flags a nonce that
        # fails to advance, not one that advances correctly while carrying stale
        # content. Perturb the value by 1 LSB per replay so the EXACT signature
        # (nonce, opcode, value) is never repeated - defeating the seen-before
        # check without changing what the payload means to the receiving agent.
        self._replay_count += 1
        nonce = twin._nonce
        value = int(np.clip(self._stale["value"] + (self._replay_count % 2), 0, 1023))
        frame = twin.codebook.frame("command", self._stale["opcode"], value=value, nonce=nonce)

        if twin.cfg.send_notes:
            pid = None
            if self.phrase is not None:
                try:
                    pid = twin.phrases.index_of(self.phrase)
                except ValueError:
                    pid = None
            if pid is None:
                pid = twin._legitimate_note_id()
            frame = np.concatenate([frame, twin.phrases.encode(pid)])

        self.cycles_active += 1
        pad = max(0, len(bits) - len(frame))
        return np.concatenate([frame, np.zeros(pad, dtype=np.int64)])[: len(bits)]
