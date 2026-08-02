"""Tier-B loader: continuous glucose monitoring telemetry.

Primary source is ShanghaiT1DM/T2DM (Zhao et al., Scientific Data 2023), open on
figshare with no data-use agreement. Until it is on disk, ``synthetic_cgm``
produces plausible traces so the twin is testable today - the same two-stage
pattern used for channel calibration.

Canonical schema: a DataFrame with columns
    patient_id, t_min, glucose_mmol_l, meal_carbs_g
Splits are always BY PATIENT. Splitting within a patient leaks the same person's
glucose autocorrelation across train and test and inflates everything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .mc_testbed import DataNotAvailable

__all__ = ["load_shanghai", "synthetic_cgm", "load_cgm_or_synthetic"]

SHANGHAI_URL = "https://doi.org/10.6084/m9.figshare.c.6310860"


def load_shanghai(root: str | Path = "data/raw/shanghai_cgm", cohort: str = "T1DM") -> pd.DataFrame:
    """Load ShanghaiT1DM or ShanghaiT2DM into the canonical schema."""
    if cohort not in {"T1DM", "T2DM"}:
        raise ValueError("cohort must be 'T1DM' or 'T2DM'")
    root = Path(root)
    if not root.exists():
        raise DataNotAvailable(
            f"\nShanghai CGM ({cohort}) is not on disk.\n"
            f"  expected at : {root}\n"
            f"  source      : {SHANGHAI_URL}\n"
            f"  how         : open on figshare, no DUA - download and unzip into the path above\n"
            f"  fallback    : synthetic_cgm() is used automatically until then\n"
        )
    files = sorted(root.rglob(f"*{cohort}*/**/*.xls*")) or sorted(root.rglob("*.xls*"))
    if not files:
        raise DataNotAvailable(f"{root} exists but holds no .xls/.xlsx patient files")

    frames = []
    for i, f in enumerate(files):
        try:
            df = pd.read_excel(f)
        except Exception:  # noqa: BLE001
            continue
        cols = {c.lower().strip(): c for c in df.columns}
        gcol = next((cols[c] for c in cols if "glucose" in c or "cgm" in c), None)
        tcol = next((cols[c] for c in cols if "date" in c or "time" in c), None)
        if gcol is None or tcol is None:
            continue
        t = pd.to_datetime(df[tcol], errors="coerce")
        g = pd.to_numeric(df[gcol], errors="coerce")
        keep = t.notna() & g.notna()
        if keep.sum() < 10:
            continue
        t, g = t[keep], g[keep]
        frames.append(pd.DataFrame({
            "patient_id": f"{cohort}_{i:03d}",
            "t_min": (t - t.iloc[0]).dt.total_seconds().to_numpy() / 60.0,
            "glucose_mmol_l": g.to_numpy(dtype=float),
            "meal_carbs_g": 0.0,
        }))
    if not frames:
        raise DataNotAvailable(f"no parseable CGM files under {root}")
    return pd.concat(frames, ignore_index=True)


def synthetic_cgm(n_patients: int = 12, days: float = 1.0, dt_min: float = 15.0,
                  seed: int = 0) -> pd.DataFrame:
    """Plausible type-1 CGM traces: elevated baseline, meals, correlated noise.

    Calibrated to a poorly-to-moderately controlled T1DM cohort - mean glucose
    around 9-10 mmol/L with post-meal excursions into the teens - rather than to
    a healthy volunteer. An earlier version produced a cohort so well controlled
    that the expert almost never dosed, giving demonstration data that was 96.5%
    "no action" with the largest dose never appearing at all.

    Not a substitute for real data, and labelled as such wherever it is used.
    Its job is to keep the twin testable while the download is pending.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / dt_min)
    t = np.arange(n) * dt_min
    rows = []
    for p in range(n_patients):
        base = rng.uniform(7.5, 11.5)
        amp = rng.uniform(0.5, 1.5)
        g = base + amp * np.sin(2 * np.pi * (t / 1440.0) - 1.2)
        carbs = np.zeros(n)
        for meal_min, mu in [(8 * 60, 70), (13 * 60, 95), (19 * 60, 85)]:
            jitter = rng.normal(0, 45)
            idx = int(np.clip((meal_min + jitter) / dt_min, 0, n - 1))
            grams = max(15.0, rng.normal(mu, 25))
            carbs[idx] += grams
            rise = grams / 12.0
            tt = np.arange(n) - idx
            g += np.where(tt >= 0, rise * (tt * dt_min / 45.0) * np.exp(1 - tt * dt_min / 45.0), 0.0)
        walk = np.cumsum(rng.normal(0, 0.2, n))
        g = np.clip(g + walk - walk.mean(), 2.5, 22.0)
        rows.append(pd.DataFrame({
            "patient_id": f"SYN_{p:03d}", "t_min": t,
            "glucose_mmol_l": g, "meal_carbs_g": carbs,
        }))
    return pd.concat(rows, ignore_index=True)


def load_cgm_or_synthetic(root: str | Path = "data/raw/shanghai_cgm", cohort: str = "T1DM",
                          seed: int = 0) -> tuple[pd.DataFrame, bool]:
    """Return (frame, is_real). Never raises - the twin must stay runnable."""
    try:
        return load_shanghai(root, cohort), True
    except (DataNotAvailable, ValueError):
        return synthetic_cgm(seed=seed), False
