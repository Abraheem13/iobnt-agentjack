"""Paired tests, multiple-comparison correction, bootstrap intervals, effect size.

Mirrors the statistical protocol used in the TierFed evaluation, for the same
reason: with a factorial design over attacks, defenses and seeds, uncorrected
p-values across a large comparison family will manufacture significance.

Two rules that are easy to break and hard to notice:

* Comparisons are PAIRED across shared seeds. Attack and defense arms run on the
  same patients with the same channel noise, so unpaired tests throw away most
  of the power and understate real effects.
* Parity is NEVER inferred from a non-significant difference. "We found no
  significant difference" is not evidence of equivalence; a one-sided
  non-inferiority test with a stated margin is.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["paired_test", "holm_bonferroni", "bca_bootstrap_ci", "hedges_g",
           "non_inferiority"]


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test plus Wilcoxon, since n is small and normality is untested."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired test needs equal-length, aligned samples")
    d = a - b
    out = {"mean_diff": float(d.mean()), "n": len(d)}
    if np.allclose(d, 0):
        return out | {"t_p": 1.0, "wilcoxon_p": 1.0}
    out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    return out


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni over the WHOLE comparison family, not per-table."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[key] = {"p_raw": float(p), "p_holm": float(adj), "significant": adj < alpha}
    return out


def bca_bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float]:
    """Bias-corrected accelerated bootstrap interval for the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    theta = x.mean()
    boots = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])

    z0 = stats.norm.ppf(np.clip((boots < theta).mean(), 1e-9, 1 - 1e-9))
    jack = np.array([np.delete(x, i).mean() for i in range(len(x))])
    jbar = jack.mean()
    denom = 6.0 * ((jbar - jack) ** 2).sum() ** 1.5
    a = ((jbar - jack) ** 3).sum() / denom if denom != 0 else 0.0

    def adj(q):
        zq = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))

    lo, hi = np.quantile(boots, [adj(alpha / 2), adj(1 - alpha / 2)])
    return float(lo), float(hi)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size with the small-sample correction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    n = len(d)
    return float(d.mean() / sd * (1 - 3 / (4 * (n - 1) - 1)))


def non_inferiority(a: np.ndarray, b: np.ndarray, margin: float) -> dict:
    """One-sided test that `a` is not worse than `b` by more than `margin`.

    Required whenever the claim is parity - for example that a defense preserves
    benign task success. A non-significant difference test does not license that
    claim and must never be reported as if it did.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return {"p": 0.0 if d.mean() + margin > 0 else 1.0, "margin": margin, "n": n}
    t = (d.mean() + margin) / se
    return {"t": float(t), "p": float(1 - stats.t.cdf(t, n - 1)), "margin": margin, "n": n}
