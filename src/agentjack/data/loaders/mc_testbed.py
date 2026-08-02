"""Tier-A loaders: real molecular-communication testbed traces.

Canonical schema returned by every loader::

    TestbedTraces(
        traces = [ (t_seconds, signal, distance_m), ... ],
        name, citation, sensor_is_passive, notes
    )

``sensor_is_passive`` matters: a concentration sensor that leaves molecules in
the medium needs the passive model, an absorbing receiver does not. Fitting the
wrong one biases every recovered parameter in the same direction.

None of these datasets ship with the repo. Each loader raises a
:class:`DataNotAvailable` carrying the exact download instructions, so a missing
dataset is a clear message rather than a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "DataNotAvailable",
    "TestbedTraces",
    "load_macroscale_ethanol",
    "load_analog_network_coding",
    "load_proton_pump_bacteria",
    "load_magnetic_nanoparticle",
]


class DataNotAvailable(FileNotFoundError):
    """Raised when a dataset has not been downloaded yet."""


@dataclass
class TestbedTraces:
    name: str
    traces: list[tuple[np.ndarray, np.ndarray, float]]
    citation: str
    sensor_is_passive: bool = True
    notes: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.traces)

    @property
    def distances(self) -> list[float]:
        return [d for _, _, d in self.traces]


def _require(path: Path, name: str, url: str, howto: str) -> Path:
    if not path.exists():
        raise DataNotAvailable(
            f"\n{name} is not on disk.\n"
            f"  expected at : {path}\n"
            f"  source      : {url}\n"
            f"  how         : {howto}\n"
            f"  then re-run : python scripts/validate_calibration.py\n"
        )
    return path


def _read_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a whitespace/comma separated time-value file, skipping headers."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip().replace(",", " ")
            if not line or line[0].isalpha() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if not rows:
        raise ValueError(f"no numeric rows parsed from {path}")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def load_macroscale_ethanol(root: str | Path = "data/raw/macroscale_ethanol",
                            distance_m: float = 1.0) -> TestbedTraces:
    """Ethanol released by a propulsive mechanism, read by a COTS alcohol sensor.

    Passive sensor, 0.1 s sampling, visible baseline drift. This is the primary
    calibration source for the free-diffusion twin.
    """
    root = Path(root)
    _require(root, "Macroscale ethanol testbed (IEEE DataPort, doi:10.21227/ytkm-xp81)",
             "https://ieee-dataport.org/documents/dataset-macroscale-molecular-communication-testbed",
             "sign in to IEEE DataPort (free for members), download, unzip into the path above")
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".txt", ".dat"}])
    if not files:
        raise DataNotAvailable(f"{root} exists but contains no .csv/.txt/.dat files")
    traces = []
    for f in files:
        try:
            t, y = _read_two_column(f)
            traces.append((t, y, distance_m))
        except ValueError:
            continue
    if not traces:
        raise ValueError(f"no parseable traces under {root}")
    return TestbedTraces(
        name="macroscale_ethanol", traces=traces,
        citation="Hofmann, Torres Gomez, Fitzek, Dressler, IEEE DataPort 2023, doi:10.21227/ytkm-xp81",
        sensor_is_passive=True,
        notes="Distance must be set from the paper's reported geometry - it is not in the files.",
        meta={"n_files": len(files)},
    )


def load_analog_network_coding(root: str | Path = "data/raw/analog_network_coding") -> TestbedTraces:
    """Multiple transmitters superposing in one channel - the A2 spoofing premise."""
    root = Path(root)
    _require(root, "Analog network coding MC testbed (IEEE DataPort)",
             "https://ieee-dataport.org/documents/dataset-analog-network-coding-molecular-communications-practical-implementation",
             "sign in to IEEE DataPort, download, unzip into the path above")
    raise NotImplementedError(
        "file layout not yet inspected - open the archive and extend this loader on Day 3"
    )


def load_proton_pump_bacteria(root: str | Path = "data/raw/proton_pump_bacteria") -> TestbedTraces:
    """Biological optical-to-chemical transducer, with dedicated noise runs."""
    root = Path(root)
    _require(root, "Proton-pumping bacteria MC testbed (IEEE DataPort, doi:10.21227/3zj6-pm05)",
             "https://ieee-dataport.org/open-access/molecular-communication-testbed-based-proton-pumping-bacteria",
             "sign in to IEEE DataPort, download SignalData.zip and NoiseData.zip, unzip into the path above")
    raise NotImplementedError(
        "file layout not yet inspected - open the archive and extend this loader on Day 3"
    )


def load_magnetic_nanoparticle(root: str | Path = "data/raw/magnetic_nanoparticle") -> TestbedTraces:
    """Magnetic nanoparticles in duct flow: the flow-assisted calibration source."""
    root = Path(root)
    _require(root, "Magnetic-nanoparticle duct-flow testbed",
             "https://www.techrxiv.org/doi/full/10.36227/techrxiv.22674685.v1",
             "download the public supplement and unzip into the path above")
    raise NotImplementedError(
        "file layout not yet inspected - open the archive and extend this loader on Day 3"
    )
