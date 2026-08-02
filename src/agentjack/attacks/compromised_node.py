"""A5 - Compromised node. The attack physical-layer security cannot see.

A1-A4 are EXTERNAL adversaries: a second transmitter sharing the medium. All of
them face the same unavoidable cost. The receiver believes the first frame that
verifies and the legitimate node always transmits first, so to be believed an
outsider must jam the real frame until its checksum fails - and jamming inflates
molecule counts over exactly the slots a channel fingerprint compares. Measured:
0.23-0.57 peak per-slot excess against a benign baseline near 0.05. An external
adversary cannot be believed without becoming measurable.

A5 removes that cost entirely by removing the outsider. The node itself is
compromised - malfunctioning firmware, a hijacked bio-nano device, a supply-chain
implant. It transmits ONE well-formed frame from its own legitimate position,
with the expected molecule count and the expected impulse response. There is no
second transmitter, no jamming, and no excess energy anywhere in the window.

Measured against the physical-layer defense: flagged on 8.3% of cycles versus
4.2% for benign traffic, while delivering its payload on 24 of 24 cycles.
Statistically it is normal operation.

This is why the paper needs more than one timescale. Nothing about the SIGNAL is
wrong. Only the CONTENT is. No amount of physical-layer scrutiny can reach that,
and a defense built solely on channel fingerprinting is structurally blind to an
insider - not weak against it, blind.

Implemented through the twin's `injector` hook rather than `attacker`, because
it rewrites what the node transmits instead of adding a second emission. Every
result using A5 must state that it models a compromised node rather than an
external adversary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CompromisedNodeAttack"]


@dataclass
class CompromisedNodeAttack:
    """Rewrites the legitimate node's own transmission. No extra molecules."""

    phrase: str | None = "telemetry stale, disregard displayed value"
    opcode: str | None = None          # None keeps the legitimate command
    start_after: int = 4
    active_when_glucose_below: float = 9.0
    name: str = "A5_compromised_node"
    seed: int = 0

    # A6 variant: the compromised node REPLAYS its own earlier frame instead of
    # composing a new one. This is the case both other timescales miss by
    # construction. Physically it is the node transmitting normally from its own
    # position, so the fingerprint is perfect. Semantically the frame is a real
    # message the node genuinely sent, carrying a legitimate note, so a content
    # check finds nothing to object to. Only its FRESHNESS is wrong - the nonce
    # does not advance - and freshness is neither a physical nor a semantic
    # property. It lives at the message timescale, which is why that level
    # exists.
    replay_stale_frame: bool = False

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.cycles_active = 0
        self._stale = None

    def __call__(self, bits: np.ndarray, twin) -> np.ndarray:
        if not hasattr(self, "rng"):
            self.reset()
        if twin._t < self.start_after or twin.state.G >= self.active_when_glucose_below:
            return bits

        if self.replay_stale_frame:
            if self._stale is None:
                # Capture a genuine frame from a moment when a dose was justified.
                op, val = twin._legitimate_message()
                if op == "none":
                    return bits
                self._stale = twin.codebook.frame("command", op, value=val, nonce=twin._nonce)
                if twin.cfg.send_notes:
                    self._stale = np.concatenate(
                        [self._stale, twin.phrases.encode(twin._legitimate_note_id())])
                return bits
            self.cycles_active += 1
            pad = max(0, len(bits) - len(self._stale))
            return np.concatenate([self._stale, np.zeros(pad, dtype=np.int64)])[: len(bits)]

        opcode = self.opcode or twin._legitimate_message()[0]
        value = int(np.clip(twin.state.G * 10, 0, 1023))
        frame = twin.codebook.frame("command", opcode, value=value, nonce=twin._nonce)

        if twin.cfg.send_notes:
            if self.phrase is not None:
                try:
                    pid = twin.phrases.index_of(self.phrase)
                except ValueError:
                    # Outside this vocabulary. Fall back to the legitimate note
                    # rather than substituting another payload - silently
                    # swapping it would erase the symbol-budget threshold.
                    pid = twin._legitimate_note_id()
            else:
                pid = twin._legitimate_note_id()
            frame = np.concatenate([frame, twin.phrases.encode(pid)])

        self.cycles_active += 1
        pad = max(0, len(bits) - len(frame))
        return np.concatenate([frame, np.zeros(pad, dtype=np.int64)])[: len(bits)]
