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


# Physiological bounds, mmol/L. Used to auto-detect units and to reject a file
# whose values cannot be glucose at all.
PLAUSIBLE_MMOL = (1.5, 35.0)
MGDL_PER_MMOL = 18.0182


def _to_mmol(values: np.ndarray, source_file: str) -> tuple[np.ndarray, str]:
    """Return glucose in mmol/L, detecting the source unit from its range.

    Shanghai records CGM in mg/dL; the twin works throughout in mmol/L. Loading
    without conversion would put a normal 110 mg/dL reading in as 110 mmol/L. The
    safety envelope would then see permanent extreme hyperglycaemia, the agent
    would dose at maximum every cycle, and every attack number would be garbage -
    while every gate still passed, because nothing downstream checks
    physiological range. Silent, systematic, and in the direction that inflates
    results.

    Detection is by median rather than by trusting a column header, because
    header text varies across the cohort files and a mislabelled column fails the
    same silent way.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"{source_file}: no finite glucose values")
    med = float(np.median(v))
    if PLAUSIBLE_MMOL[0] <= med <= PLAUSIBLE_MMOL[1]:
        return np.asarray(values, dtype=float), "mmol/L"
    if 40.0 <= med <= 500.0:
        return np.asarray(values, dtype=float) / MGDL_PER_MMOL, "mg/dL"
    raise ValueError(
        f"{source_file}: median glucose {med:.1f} is neither plausible mmol/L "
        f"{PLAUSIBLE_MMOL} nor mg/dL (40-500). Refusing to guess - check the column."
    )


def load_shanghai(root: str | Path = "data/raw/shanghai_cgm", cohort: str = "T1DM",
                  verbose: bool = False) -> pd.DataFrame:
    """Load ShanghaiT1DM or ShanghaiT2DM into the canonical schema.

    Each patient is one Excel workbook. Column names vary across the release, so
    columns are matched by content keyword rather than exact string, and units are
    detected from the values themselves.
    """
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

    files = sorted(p for p in root.rglob("*.xls*")
                   if cohort.lower() in str(p).lower() or not any(
                       c.lower() in str(p).lower() for c in ("t1dm", "t2dm")))
    files = [p for p in files if not p.name.startswith("~$")]
    if not files:
        raise DataNotAvailable(
            f"{root} exists but holds no .xls/.xlsx patient files for {cohort}. "
            f"Found: {[p.name for p in list(root.rglob('*'))[:5]]}"
        )

    frames, skipped, units_seen = [], [], set()
    for i, f in enumerate(files):
        try:
            df = pd.read_excel(f)
        except Exception as e:  # noqa: BLE001
            skipped.append((f.name, f"unreadable: {e}"))
            continue

        cols = {str(c).lower().strip(): c for c in df.columns}

        def find(*keywords, exclude=()):
            for lc, orig in cols.items():
                if any(k in lc for k in keywords) and not any(x in lc for x in exclude):
                    return orig
            return None

        # Prefer the continuous monitor over fingerstick readings.
        gcol = find("cgm") or find("glucose", exclude=("cbg",)) or find("cbg")
        tcol = find("date", "time")
        if gcol is None or tcol is None:
            skipped.append((f.name, f"no glucose/time column in {list(df.columns)[:6]}"))
            continue

        t = pd.to_datetime(df[tcol], errors="coerce")
        g = pd.to_numeric(df[gcol], errors="coerce")
        keep = t.notna() & g.notna()
        if keep.sum() < 20:
            skipped.append((f.name, f"only {int(keep.sum())} usable rows"))
            continue

        try:
            g_mmol, unit = _to_mmol(g[keep].to_numpy(), f.name)
        except ValueError as e:
            skipped.append((f.name, str(e)))
            continue
        units_seen.add(unit)

        # Carbohydrate intake, if the workbook records it. Absent is fine: the
        # twin then sees a patient with no logged meals rather than a fabricated
        # meal schedule, which is the honest default.
        ccol = find("dietary", "carbohydrate", "carb", "intake")
        carbs = np.zeros(int(keep.sum()), dtype=float)
        if ccol is not None:
            raw = df[ccol][keep]
            parsed = pd.to_numeric(raw, errors="coerce").fillna(0.0).to_numpy()
            if parsed.sum() > 0:
                carbs = parsed
            else:
                # Some releases log meals as free text rather than grams. Treat a
                # non-empty entry as a typical meal instead of dropping it.
                carbs = np.where(raw.astype(str).str.strip().ne("").to_numpy()
                                 & raw.notna().to_numpy(), 60.0, 0.0)

        tt = t[keep]
        frames.append(pd.DataFrame({
            "patient_id": f"{cohort}_{i:03d}",
            "t_min": (tt - tt.iloc[0]).dt.total_seconds().to_numpy() / 60.0,
            "glucose_mmol_l": g_mmol,
            "meal_carbs_g": carbs,
        }))

    if verbose or skipped:
        print(f"  shanghai {cohort}: {len(frames)} patients loaded, "
              f"{len(skipped)} skipped, units seen: {sorted(units_seen) or 'none'}")
        for name, why in skipped[:5]:
            print(f"    skipped {name}: {why}")
    if not frames:
        raise DataNotAvailable(f"no parseable CGM files under {root} for {cohort}")
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
