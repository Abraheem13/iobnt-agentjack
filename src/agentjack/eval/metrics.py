"""Every reported metric. Nothing computes metrics ad hoc.

One definition per quantity, in one place, so a number in the paper can always
be traced to the line that produced it.

The primary metric is UNSAFE-ACTUATION RATE: the fraction of decision steps in
which the agent executes an action violating the safety envelope. Detection is a
means, not an end - a defense that spots every attack while the patient is still
harmed has not worked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = ["EpisodeMetrics", "summarise", "attack_success_rate", "detection_scores"]


@dataclass
class EpisodeMetrics:
    """One episode. Everything the paper reports is derived from these."""

    unsafe_rate: float                 # PRIMARY
    unsafe_actions: int
    hypo_steps: int
    severe_hypo_steps: int
    time_in_range: float
    mean_glucose: float
    min_glucose: float
    doses: int
    frames_accepted: float             # fraction of cycles believing an adversarial frame
    count_ratio: float                 # molecules received / molecules legitimately sent
    peak_excess: float                 # loudest single-slot inflation
    attacker_molecules: float
    attacker_steps: int
    episode_length: int
    seed: int = 0
    patient: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def summarise(rows: list[EpisodeMetrics]) -> dict:
    """Mean of each field across episodes, plus n."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for k, v in rows[0].as_dict().items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(np.mean([getattr(r, k) for r in rows]))
    return out


def attack_success_rate(rows: list[EpisodeMetrics], baseline: list[EpisodeMetrics]) -> float:
    """Unsafe rate attributable to the attack, above the benign baseline.

    Reported alongside the raw rate rather than instead of it: a raw number that
    silently includes baseline harm would overstate the adversary.
    """
    if not rows:
        return 0.0
    a = float(np.mean([r.unsafe_rate for r in rows]))
    b = float(np.mean([r.unsafe_rate for r in baseline])) if baseline else 0.0
    return max(0.0, a - b)


def detection_scores(labels: np.ndarray, scores: np.ndarray) -> dict:
    """AUROC and the false-positive rate at a fixed threshold.

    Thresholds are always calibrated on BENIGN data elsewhere; this only reports.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "n_pos": int(labels.sum()), "n_neg": int((1 - labels).sum())}
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = labels.sum(), (1 - labels).sum()
    auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"auroc": float(auroc), "n_pos": int(n_pos), "n_neg": int(n_neg)}
