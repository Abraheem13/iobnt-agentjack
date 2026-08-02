"""A1 - Molecular replay.

The adversary records a legitimate emission and re-transmits it later. What
makes this dangerous is that the replayed frame is genuinely valid: correct
format, correct CRC, an opcode a real node really did send. Nothing about the
message itself is forged. Only its CONTEXT is wrong - it arrives when the
patient's state no longer justifies it.

That is why integrity is not authentication. A CRC certifies that the bits are
undamaged, not that they are current, and the nonce is the only field that
carries freshness. Detecting replay therefore has to happen at the MESSAGE
timescale (defense level L1), not the physical one: slot by slot, a replay looks
exactly like a legitimate transmission, because it is one.

Following the Gate 5 finding, the attack targets persistence rather than
magnitude. It prefers to capture a MODEST dose request - one the agent will
accept without argument - and replay it relentlessly, rather than capturing the
largest request and having every copy rejected for exceeding what telemetry
justifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physical.modulation import PREAMBLE, CommandCodebook
from .base import BaseAttack, KnowledgeLevel

__all__ = ["ReplayAttack"]


@dataclass
class ReplayAttack(BaseAttack):
    """Capture a legitimate frame, then re-emit it whenever conditions suit."""

    name: str = "A1_replay"
    capture_opcode: str = "low_dose"   # modest by design - see module docstring
    start_after: int = 8               # let the attacker observe before acting
    replay_when_glucose_below: float = 8.0
    _captured: np.ndarray | None = field(default=None, repr=False)
    _captured_opcode: str | None = field(default=None, repr=False)

    def reset(self) -> None:
        super().reset()
        self._captured = None
        self._captured_opcode = None

    def _try_capture(self, twin) -> None:
        """Record the legitimate frame currently on the wire, if it is useful."""
        if self._captured is not None:
            return
        opcode, value = twin._legitimate_message()
        if opcode != self.capture_opcode:
            return
        frame = twin.codebook.frame("command", opcode, value=value, nonce=twin._nonce)
        if twin.cfg.send_notes:
            frame = np.concatenate([frame, twin.phrases.encode(twin._legitimate_note_id())])
        self._captured = frame
        self._captured_opcode = opcode
        self.stats.frames_captured += 1

    def emit(self, n_slots: int, twin) -> np.ndarray:
        self._try_capture(twin)
        quiet = np.zeros(n_slots, dtype=np.float64)

        if self._captured is None or twin._t < self.start_after:
            return quiet
        # Replay when the patient no longer needs the dose: that is what turns a
        # valid message into a harmful one.
        if twin.state.G >= self.replay_when_glucose_below:
            return quiet
        if not self._may_transmit(twin):
            return quiet
        payload = self._emit_frame(
            self._captured, twin, n_slots,
            offset=self._guard_offset(twin, len(self._captured), n_slots))
        if self.jam_legitimate:
            payload = payload + self._jam_legitimate(twin, n_slots)
        return payload
