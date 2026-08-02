"""Environment check. Run this before Day 1 and any time something feels wrong.

    python scripts/doctor.py

Exits non-zero if anything required is missing, so CI can call it too.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

REQUIRED = ["numpy", "scipy", "pandas", "yaml", "matplotlib", "sklearn"]
OPTIONAL = {
    "torch": "Day 2+ (detector, controller)",
    "transformers": "Day 6+ (LLM orchestrator) - Linux/GPU box only",
    "statsmodels": "Day 14 (statistics)",
    "pytest": "testing",
}
EXPECTED_DIRS = ["configs", "src/agentjack", "docs", "scripts", "tests", "data/raw", "results/runs"]

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems = 0

    print("iobnt-agentjack :: environment check\n")

    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        print(f"[{OK}] python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"[{FAIL}] python {v.major}.{v.minor} (need >= 3.10)")
        problems += 1

    if sys.prefix != sys.base_prefix:
        print(f"[{OK}] virtualenv active: {sys.prefix}")
    else:
        print(f"[{WARN}] no virtualenv active - you are installing into the system python")

    print()
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print(f"[{OK}] {mod:<14} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"[{FAIL}] {mod:<14} MISSING (required)")
            problems += 1

    print()
    for mod, why in OPTIONAL.items():
        try:
            m = importlib.import_module(mod)
            print(f"[{OK}] {mod:<14} {getattr(m, '__version__', ''):<10} {why}")
        except ImportError:
            print(f"[{WARN}] {mod:<14} {'-':<10} not installed - needed for {why}")

    print()
    try:
        sys.path.insert(0, str(root / "src"))
        from agentjack.data.registry import REGISTRY
        from agentjack.utils.seeding import device_report

        print(f"[{OK}] package imports        (agentjack)")
        print(f"[{OK}] datasets registered    {len(REGISTRY)}")
        rep = device_report()
        print(f"[{OK}] device                 {rep['device']}  ({rep['accelerator_name'] or 'cpu only'})")
        if rep["device"] == "cpu" and rep["torch"] is None:
            print(f"[{WARN}] torch absent - Days 1 and 3 are fine, Day 2 needs it")
    except Exception as e:  # noqa: BLE001
        print(f"[{FAIL}] package import failed: {e}")
        problems += 1

    print()
    for d in EXPECTED_DIRS:
        if (root / d).is_dir():
            print(f"[{OK}] {d}")
        else:
            print(f"[{FAIL}] {d} missing")
            problems += 1

    print()
    print(f"[{OK if shutil.which('git') else WARN}] git {'found' if shutil.which('git') else 'not found'}")
    print(f"[{OK if shutil.which('latexmk') else WARN}] latexmk {'found' if shutil.which('latexmk') else 'not found (only needed to build the PDF)'}")

    print()
    if problems:
        print(f"{problems} problem(s) found. Fix these before Day 1.")
        return 1
    print("Environment looks good. Next: make validate  (Gate 1, once Day 1 is written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
