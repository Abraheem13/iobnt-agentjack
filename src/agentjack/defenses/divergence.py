"""Cross-timescale divergence and threshold calibration for the nested monitor.

Each level of the monitor watches a different rate of change, and they cannot be
compared on a common scale by construction: a per-slot count residual and an
episode-level behavioural drift are not the same kind of number. Fusing raw
scores would let whichever level happens to have the largest units dominate.

Levels are therefore converted to a common currency first - how far this cycle's
score sits from the level's own BENIGN distribution, in that distribution's own
units. A level that has never seen anything unusual contributes nothing, however
large its raw score.

That normalisation is also what lets the fused score keep a stated
false-positive rate. Calibration uses benign runs only; see defenses/base.py for
why that discipline is not negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LevelCalibration", "robust_z", "fuse"]


def robust_z(x: float, median: float, mad: float) -> float:
    """Distance from the benign median in robust standard deviations.

    Median and MAD rather than mean and standard deviation because benign score
    distributions here are skewed and occasionally spiky - a single noisy cycle
    would otherwise inflate the scale and blind the level for the rest of the run.
    """
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return 0.0 if x <= median else float("inf")
    return float((x - median) / scale)


@dataclass
class LevelCalibration:
    """The benign distribution of one level's score."""

    median: float = 0.0
    mad: float = 0.0
    n: int = 0
    all_zero: bool = False

    @classmethod
    def fit(cls, scores: np.ndarray) -> "LevelCalibration":
        s = np.asarray(scores, dtype=float)
        if s.size == 0:
            return cls()
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        return cls(median=med, mad=mad, n=int(s.size), all_zero=bool(np.allclose(s, 0.0)))

    def normalise(self, x: float) -> float:
        """Benign-relative deviation, floored at zero.

        Only EXCESS matters: a level scoring below its benign median is not
        evidence of an attack, and letting it contribute a negative value would
        allow one quiet level to mask another that is alarming.
        """
        if self.all_zero:
            # The level never fires on benign traffic, so any positive score is
            # already anomalous and needs no scale.
            return float(max(0.0, x) * 10.0)
        return max(0.0, robust_z(x, self.median, self.mad))


def fuse(deviations: dict[int, float], weights: dict[int, float] | None = None) -> float:
    """Combine per-level deviations into one trust score.

    A weighted MAXIMUM, not a mean. The levels cover disjoint attacks - the
    physical level sees external jamming and is blind to a compromised node,
    the semantic level the reverse - so averaging would dilute a confident
    detection by one level with the silence of the others that cannot possibly
    see this attack. Taking the maximum means a level is believed when it speaks.
    """
    if not deviations:
        return 0.0
    w = weights or {}
    return float(max(w.get(k, 1.0) * v for k, v in deviations.items()))
