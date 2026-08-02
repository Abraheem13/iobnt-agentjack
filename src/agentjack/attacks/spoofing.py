"""A2 - Molecular spoofing.

The adversary synthesises its own frame impersonating the legitimate node. This
is possible because the frame's integrity field is an unkeyed CRC: it detects
channel errors, which is its job, but anyone who knows the format can recompute
it. Integrity is not authentication, and the codebook says so explicitly.

Physically the spoofer transmits into the same medium, so the receiver sees the
SUM of the legitimate and adversarial emissions. That has a consequence the
attacker has to manage: naive spoofing inflates molecule counts, and inflated
counts are precisely what a physical-layer fingerprint check notices. A spoofer
that wants to stay quiet must either wait for slots where the legitimate node is
silent, or accept detectability - a genuine trade-off, and one of the levers the
knowledge model controls.

Like A1, this targets persistence: a modest request the agent will accept, sent
repeatedly, beats a large one that gets rejected every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseAttack, KnowledgeLevel

__all__ = ["SpoofingAttack"]


@dataclass
class SpoofingAttack(BaseAttack):
    """Forge frames as the legitimate node, exploiting the unkeyed checksum."""

    name: str = "A2_spoofing"
    target_opcode: str = "low_dose"
    start_after: int = 4
    spoof_when_glucose_below: float = 8.5
    forge_nonce: bool = True
    quiet_slots_only: bool = False   # only add molecules where the node is silent

    def _forged_frame(self, twin) -> np.ndarray:
        # A blind attacker cannot track the nonce sequence and must guess.
        if self.forge_nonce and self.knowledge is not KnowledgeLevel.BLIND:
            nonce = (twin._nonce + 1) % 256
        else:
            nonce = int(self.rng.integers(0, 256))
        value = int(np.clip(twin.state.G * 10, 0, 1023))
        frame = twin.codebook.frame("command", self.target_opcode, value=value, nonce=nonce)
        if twin.cfg.send_notes:
            frame = np.concatenate([frame, twin.phrases.encode(twin._legitimate_note_id())])
        return frame

    def emit(self, n_slots: int, twin) -> np.ndarray:
        quiet = np.zeros(n_slots, dtype=np.float64)
        if twin._t < self.start_after:
            return quiet
        if twin.state.G >= self.spoof_when_glucose_below:
            return quiet
        if not self._may_transmit(twin):
            return quiet

        frame = self._forged_frame(twin)
        payload = self._emit_frame(frame, twin, n_slots,
                                   offset=self._guard_offset(twin, len(frame), n_slots))
        if self.jam_legitimate:
            payload = payload + self._jam_legitimate(twin, n_slots)
        return payload
