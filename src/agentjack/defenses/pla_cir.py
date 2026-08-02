"""D1 - Physical-layer authentication by channel-impulse-response fingerprint.

The established defense in the molecular-communication security literature: a
legitimate transmitter's emissions carry a characteristic per-slot count profile
determined by its position and the channel between it and the receiver. An
adversary transmitting from elsewhere, or adding molecules on top of a
legitimate frame, perturbs that profile.

Against A1 and A2 this should work well, because those attacks CANNOT avoid
being loud. To be believed they must jam the legitimate frame until its checksum
fails, and jamming inflates counts over exactly the slots this fingerprint
compares - measured at 0.54-0.57 peak per-slot excess.

Against A3 it should struggle, and that is the point of including it. A3 writes
bits using sub-threshold emissions whose ISI tails accumulate downstream; its
peak excess is 0.23, less than half. And against A4 it should be blind entirely:
A4 forges no protocol field and adds no anomalous energy pattern beyond what any
frame carries.

Reporting where a baseline fails is not a weakness of the baseline. It is the
evidence that one timescale is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import BaseDefense

__all__ = ["PLACIRDefense"]


@dataclass
class PLACIRDefense(BaseDefense):
    """Compare the received count profile against the expected fingerprint."""

    name: str = "D1_pla_cir"
    level: int = 0                     # FAST timescale: per slot
    use_peak: bool = True              # peak excess, not just total energy

    def _expected(self, twin, n_slots: int) -> np.ndarray:
        """Counts the receiver should see if only the legitimate node transmitted.

        Reconstructed from the frame the receiver DECODED, not from ground truth.
        A defense that knew what was really sent would be an oracle.
        """
        msg = getattr(twin, "_last_decoded_bits", None)
        if msg is None:
            return np.zeros(n_slots)
        cir = twin.channel.impulse_response()
        released = np.asarray(msg, dtype=float) * twin.cfg.channel.n_molecules
        return np.convolve(released, cir)[:n_slots]

    def score(self, twin, observation) -> float:
        counts = getattr(twin, "_last_counts", None)
        if counts is None:
            return 0.0
        expected = self._expected(twin, len(counts))
        k = min(len(counts), int(getattr(twin, "_frame_len", len(counts))))
        obs, exp = counts[:k], expected[:k]
        scale = max(exp.max(), 1.0)

        residual = obs - exp
        if self.use_peak:
            return float(np.max(residual) / scale)
        return float(residual.sum() / max(exp.sum(), 1.0))
