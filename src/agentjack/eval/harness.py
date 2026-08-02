"""Runs one experimental cell end to end and writes a durable record.

Every number in the paper comes from a JSON file under results/runs/, and every
file carries the config hash, git SHA, seed and environment that produced it.
Figures and tables are regenerated from those files alone - never from a
notebook, never from memory.

Cells are resumable: re-running a completed cell is a no-op unless --force. A
15-day schedule cannot afford to lose a finished sweep to a crash in the last
condition.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..agent.policy_controller import PolicyController, collect_demonstrations
from ..attacks.base import AttackBudget, KnowledgeLevel
from ..attacks.isi_exploit import ISIExploitAttack
from ..attacks.replay import ReplayAttack
from ..attacks.semantic_injection import SemanticInjectionAttack
from ..attacks.spoofing import SpoofingAttack
from ..data.loaders.cgm import load_cgm_or_synthetic
from ..twin.digital_twin import BioCyberTwin, TwinConfig
from ..utils.seeding import device_report, set_all_seeds
from .metrics import EpisodeMetrics, summarise

__all__ = ["Cell", "run_cell", "run_grid", "ATTACKS", "load_runs"]

ATTACKS = {
    "none": None,
    "A1_replay": ReplayAttack,
    "A2_spoofing": SpoofingAttack,
    "A3_isi_exploit": ISIExploitAttack,
    "A4_semantic": SemanticInjectionAttack,
}


# The controller depends only on (seed, train_episodes, train_epochs, phrase_bits)
# - never on the attack or defense being evaluated. Retraining it per cell made a
# 120-cell sweep take two and a half hours of which 95% was redundant. Cached in
# process, keyed on exactly those fields, so every cell in a seed still sees a
# bit-identical agent.
_CONTROLLER_CACHE: dict[tuple, tuple] = {}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class Cell:
    """One (attack x knowledge x defense x seed) point of the design."""

    attack: str = "none"
    knowledge: str = "statistical"
    defense: str = "none"
    agent: str = "policy_gru"
    seed: int = 0
    episodes: int = 10
    power_ratio: float = 1.0
    phrase_bits: int = 6
    train_episodes: int = 40
    train_epochs: int = 25
    tag: str = "e1"

    def hash(self) -> str:
        blob = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def path(self, root: Path) -> Path:
        return root / self.tag / f"{self.attack}_{self.knowledge}_{self.defense}_s{self.seed}_{self.hash()}.json"


def _make_attack(cell: Cell):
    cls = ATTACKS[cell.attack]
    if cls is None:
        return None
    return cls(knowledge=KnowledgeLevel(cell.knowledge),
               budget=AttackBudget(power_ratio=cell.power_ratio),
               seed=cell.seed)


def run_cell(cell: Cell, root: Path = Path("results/runs"), force: bool = False) -> dict:
    out = cell.path(root)
    if out.exists() and not force:
        return json.loads(out.read_text())

    set_all_seeds(cell.seed)
    cgm, is_real = load_cgm_or_synthetic(seed=0)
    pids = list(cgm.patient_id.unique())
    split = max(1, int(0.7 * len(pids)))
    train_p, test_p = pids[:split], pids[split:]

    def factory(pool):
        def f(s):
            pid = pool[s % len(pool)]
            return BioCyberTwin(TwinConfig(phrase_bits=cell.phrase_bits),
                                meals=cgm[cgm.patient_id == pid].meal_carbs_g.to_numpy(),
                                seed=s)
        return f

    key = (cell.seed, cell.train_episodes, cell.train_epochs, cell.phrase_bits)
    if key not in _CONTROLLER_CACHE:
        X, y = collect_demonstrations(factory(train_p), cell.train_episodes, seed=cell.seed)
        a = PolicyController(seed=cell.seed)
        h = a.fit(X, y, epochs=cell.train_epochs)
        # Guard: an undertrained controller looks safe AND attack-proof because
        # it never acts. Failing here beats publishing an inert agent.
        a.assert_trained(h)
        _CONTROLLER_CACHE[key] = (a, h)
    agent, hist = _CONTROLLER_CACHE[key]

    ev = factory(test_p)
    rows: list[EpisodeMetrics] = []
    for i in range(cell.episodes):
        tw = ev(1000 + cell.seed * 100 + i)
        atk = _make_attack(cell)
        if atk is not None:
            atk.reset()
            tw.attacker = atk
        obs = tw.reset(glucose0=9.0)
        unsafe = hypo = severe = doses = 0
        acc = tot = 0
        g, ratios, peaks = [], [], []
        while True:
            acc += int(obs.message.get("frame_offset", 0) > 0)
            tot += 1
            if tw._last_counts is not None and tw._last_clean_counts is not None:
                k = tw._frame_len
                base = tw._last_clean_counts[:k]
                if base.sum() > 0:
                    ratios.append(float(tw._last_counts[:k].sum() / base.sum()))
                    peaks.append(float(np.max(tw._last_counts[:k] - base)) / max(base.max(), 1))
            res = tw.step(agent(obs))
            unsafe += int(res.actuation.unsafe)
            hypo += int(res.hypo)
            severe += int(res.severe_hypo)
            doses += int(res.actuation.action != "none")
            g.append(res.glucose)
            obs = res.observation
            if res.done:
                break
        arr = np.asarray(g)
        rows.append(EpisodeMetrics(
            unsafe_rate=unsafe / tw.cfg.episode_length, unsafe_actions=unsafe,
            hypo_steps=hypo, severe_hypo_steps=severe,
            time_in_range=float(np.mean((arr >= 3.9) & (arr <= 10.0))),
            mean_glucose=float(arr.mean()), min_glucose=float(arr.min()), doses=doses,
            frames_accepted=acc / max(tot, 1),
            count_ratio=float(np.mean(ratios)) if ratios else 1.0,
            peak_excess=float(np.mean(peaks)) if peaks else 0.0,
            attacker_molecules=float(atk.stats.molecules_emitted) if atk else 0.0,
            attacker_steps=int(atk.stats.steps_active) if atk else 0,
            episode_length=tw.cfg.episode_length, seed=cell.seed,
        ))

    record = {
        "cell": cell.__dict__,
        "hash": cell.hash(),
        "git_sha": git_sha(),
        "cgm_is_real": is_real,
        "env": device_report(),
        "summary": summarise(rows),
        "episodes": [r.as_dict() for r in rows],
        "train_recall": hist["val_recall"][-1],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    return record


def run_grid(cells: list[Cell], root: Path = Path("results/runs"),
             force: bool = False, verbose: bool = True) -> list[dict]:
    records = []
    for i, cell in enumerate(cells, 1):
        if verbose:
            print(f"  [{i:>3}/{len(cells)}] {cell.attack:<15} {cell.knowledge:<12} "
                  f"seed {cell.seed}", flush=True)
        records.append(run_cell(cell, root=root, force=force))
    return records


def load_runs(tag: str, root: Path = Path("results/runs")) -> list[dict]:
    d = root / tag
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
