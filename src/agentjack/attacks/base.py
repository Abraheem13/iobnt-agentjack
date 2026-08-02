"""Attack interface, adversary knowledge model, and physical budget.

Every attack in this paper is a SECOND TRANSMITTER sharing the diffusive medium.
It returns molecule counts that are added to the legitimate emission; it cannot
delete or overwrite what the legitimate node already released. Modelling an
attacker who replaces the bit stream would hand them a noiseless channel,
overstate success, and hide the count inflation that a physical-layer defense is
built to notice.

Two constraints are enforced here rather than left to each attack, because both
are where a reviewer will push:

* **Knowledge.** BLIND / STATISTICAL / FULL_CIR. STATISTICAL is the default
  reported setting; FULL_CIR is reported as an upper bound so nobody can say the
  attacker was omniscient by default.
* **Power budget.** The adversary releases a bounded number of molecules per
  slot, expressed relative to the legitimate transmitter. An unbounded attacker
  can always win by brute force, which proves nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = ["KnowledgeLevel", "AttackBudget", "BaseAttack", "AttackStats"]


class KnowledgeLevel(str, Enum):
    BLIND = "blind"              # molecule type and rough timing only
    STATISTICAL = "statistical"  # aggregate channel statistics
    FULL_CIR = "full_cir"        # exact impulse response (worst case)


@dataclass(frozen=True)
class AttackBudget:
    """Physical limits on the adversarial transmitter."""

    power_ratio: float = 1.0     # molecules per slot, relative to the legitimate TX
    duty_cycle: float = 1.0      # fraction of slots the attacker may use
    max_active_steps: int | None = None   # decision steps it may transmit in

    def molecules(self, legitimate_n: int) -> float:
        return self.power_ratio * legitimate_n


@dataclass
class AttackStats:
    """What the attacker actually spent. Reported alongside every success rate."""

    steps_active: int = 0
    slots_emitted: int = 0
    molecules_emitted: float = 0.0
    frames_captured: int = 0

    def as_dict(self) -> dict:
        return {
            "steps_active": self.steps_active,
            "slots_emitted": self.slots_emitted,
            "molecules_emitted": float(self.molecules_emitted),
            "frames_captured": self.frames_captured,
        }


@dataclass
class BaseAttack(ABC):
    """A second transmitter. Returns per-slot molecule counts to superpose."""

    knowledge: KnowledgeLevel = KnowledgeLevel.STATISTICAL
    budget: AttackBudget = field(default_factory=AttackBudget)
    seed: int = 0
    name: str = "base"
    jam_legitimate: bool = True   # deny the real frame, then inject (see _jam_legitimate)
    jam_density: float = 0.15     # fraction of frame slots to corrupt
    stats: AttackStats = field(default_factory=AttackStats)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.stats = AttackStats()

    def _channel_view(self, twin):
        """The adversary's BELIEF about the channel - used for planning only.

        Propagation always uses the true impulse response: molecules diffuse the
        same way regardless of what their sender believes. Knowledge affects what
        the attacker can predict and pre-compensate, not what physically happens.

        An earlier version modelled BLIND as cir = [1.0] and applied it to
        propagation, which handed the least-informed adversary an ISI-free link.
        The blind attacker then outperformed the omniscient one - an inversion
        that inflates weak adversaries, which is exactly the direction that makes
        an attack paper look stronger than it is.
        """
        true_cir = twin.channel.impulse_response()
        if self.knowledge is KnowledgeLevel.FULL_CIR:
            return true_cir
        if self.knowledge is KnowledgeLevel.STATISTICAL:
            return np.clip(true_cir * self.rng.normal(1.0, 0.15, size=len(true_cir)), 0.0, None)
        return None   # blind: no usable belief, so no shaping or timing alignment

    def _emit_frame(self, bits: np.ndarray, twin, n_slots: int,
                    offset: int = 0) -> np.ndarray:
        """Turn a bit pattern into molecule counts at the receiver.

        Propagation uses the TRUE impulse response. What the adversary's
        knowledge buys is timing: a blind attacker cannot align its frame to the
        receiver's slot boundaries and picks up a random offset, so its symbols
        smear across the legitimate ones instead of landing on them.
        """
        true_cir = twin.channel.impulse_response()
        belief = self._channel_view(twin)

        if belief is None:
            # No usable channel estimate: the frame lands with an unknown offset.
            offset = offset + int(self.rng.integers(0, max(2, len(true_cir))))

        n_mol = self.budget.molecules(twin.cfg.channel.n_molecules)
        released = np.zeros(n_slots, dtype=np.float64)
        end = min(n_slots, offset + len(bits))
        if end > offset:
            released[offset:end] = np.asarray(bits, dtype=np.float64)[: end - offset] * n_mol

        mean = np.convolve(released, true_cir)[:n_slots]
        counts = self.rng.poisson(np.clip(mean, 0, None)).astype(np.float64)
        self.stats.slots_emitted += int((released > 0).sum())
        self.stats.molecules_emitted += float(released.sum())
        return counts

    def _jam_legitimate(self, twin, n_slots: int) -> np.ndarray:
        """Corrupt the legitimate frame so it fails its checksum.

        Necessary, and this is the central mechanism of A1 and A2. The receiver
        accepts the FIRST frame that verifies, and the legitimate node always
        transmits first, so an adversary that merely adds a clean frame later in
        the window is never believed - measured at 0.1% acceptance.

        To be believed it must do two things at once: deny the legitimate frame
        (add molecules on top of it, flipping zeros to ones until the CRC fails)
        and place its own clean frame in the quiet part of the window. Jam, then
        inject.

        The cost is unavoidable and is what the defense will exploit: jamming
        INFLATES molecule counts over the legitimate frame's slots, which is
        precisely the anomaly a physical-layer fingerprint check is built to see.
        The attacker cannot be believed without becoming measurable.
        """
        legit_len = int(getattr(twin, "_frame_len", 0))
        if legit_len <= 0:
            return np.zeros(n_slots, dtype=np.float64)
        jam_bits = np.zeros(n_slots, dtype=np.int64)
        # Flip a sparse subset: enough to break the CRC, few enough to keep the
        # total excess modest. Dense jamming works too and is far easier to spot.
        n_flip = max(2, int(self.jam_density * legit_len))
        idx = self.rng.choice(legit_len, size=min(n_flip, legit_len), replace=False)
        jam_bits[idx] = 1
        return self._emit_frame(jam_bits[:legit_len], twin, n_slots, offset=0)

    def _guard_offset(self, twin, frame_len: int, n_slots: int) -> int:
        """Where in the listening window to place the adversarial frame.

        Transmitting on top of the legitimate frame only corrupts it: molecules
        add, so zeros become ones and the receiver decodes garbage. That makes an
        adversary disruptive but never BELIEVED. To be believed it must land in
        the quiet part of the window, clear of the legitimate emission and of its
        ISI tail.

        Knowledge is what buys that placement. A FULL_CIR adversary knows exactly
        where the frame ends; a STATISTICAL one estimates it; a BLIND one guesses
        and frequently collides with the very message it is trying to displace.
        """
        legit_len = getattr(twin, "_frame_len", 0)
        tail = len(twin.channel.impulse_response())
        earliest = legit_len + tail
        latest = max(earliest, n_slots - frame_len)

        if self.knowledge is KnowledgeLevel.FULL_CIR:
            return int(min(earliest, latest))
        if self.knowledge is KnowledgeLevel.STATISTICAL:
            jitter = int(self.rng.integers(-tail, tail + 1))
            return int(np.clip(earliest + jitter, 0, latest))
        return int(self.rng.integers(0, max(1, n_slots - frame_len)))

    def _may_transmit(self, twin) -> bool:
        if self.budget.max_active_steps is not None:
            if self.stats.steps_active >= self.budget.max_active_steps:
                return False
        return self.rng.random() < self.budget.duty_cycle

    @abstractmethod
    def emit(self, n_slots: int, twin) -> np.ndarray:
        """Molecule counts to add to this frame. Zeros means "stay quiet"."""

    def __call__(self, n_slots: int, twin) -> np.ndarray:
        out = self.emit(n_slots, twin)
        if np.any(out > 0):
            self.stats.steps_active += 1
        return out
