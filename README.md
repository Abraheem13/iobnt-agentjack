# IoBNT-AgentJack

**Molecular prompt injection against agentic bio-cyber interfaces, and a nested
multi-timescale trust monitor for the Internet of Bio-Nano Things.**

> Target venue: IEEE TMBMC Special Issue — *Agentic AI for Autonomous Bio-Cyber
> Interfaces in Molecular Communications and IoBNT* (deadline 30 Dec 2026).
> Status: **pre-alpha, under active 15-day construction.**

---

## What this is

When an AI agent orchestrates a bio-cyber gateway — reading decoded molecular
messages and biosensor telemetry, then issuing actuation commands — the molecular
channel becomes an untrusted input to a decision-making agent. This repo shows
that an adversarial nanotransmitter can exploit that: crafted molecular emissions
that, after diffusion and inter-symbol interference, are decoded into content
that hijacks the agent's decisions.

We call this **molecular prompt injection**, and we defend against it with a
**nested multi-timescale trust monitor** that checks physical, message and
semantic consistency at different update frequencies and fuses them by
divergence.

## Why nesting is the point

Different attacks betray themselves on different timescales:

| Timescale | Update rate | Catches |
|---|---|---|
| **L0 physical** | every slot | spoofing, anomalous emissions |
| **L1 message** | every message | replay, stale or forged sequences |
| **L2 semantic** | every episode | instruction hijacking, behavioural drift |

An ISI-exploiting attack looks unremarkable slot by slot. A semantic injection
arrives over a physically clean channel. Neither single-timescale defense sees
both — which is the ablation the paper turns on.

## Repository layout

```
configs/      every experimental variable, nothing hard-coded
  channel/    diffusion, flow, testbed-calibrated, severity sweeps
  agent/      learned controller (primary) and LLM orchestrator
  attack/     A1 replay . A2 spoofing . A3 ISI-exploit . A4 semantic . A5 adaptive
  defense/    none . PLA-CIR . LLM guardrail . nested monitor . ablations
  task/       glycemic control . sepsis alerting
  experiment/ E1-E6, the six runs that make the paper
src/agentjack/
  channel/    MC physics + closed-form validation + hardware calibration
  physical/   modulation, detection, CIR fingerprinting
  twin/       digital twin, biology, actuation + safety envelope
  agent/      agent interface, LLM orchestrator, learned controller, tools
  attacks/    the attack suite and the payload/symbol-budget machinery
  defenses/   baselines, the nested monitor, divergence fusion, attribution
  data/       registry (single source of truth) + per-dataset loaders
  eval/       harness, metrics, statistics, ablations
docs/         charter, threat model, datasets, protocol, 15-day plan
paper/        LaTeX source
```

## Quickstart

```bash
make setup                 # editable install + hooks
make validate              # GATE 1: simulator physics vs closed form
make data                  # fetch registered datasets into data/raw
make run CFG=configs/experiment/e1_attack_success.yaml
make figures tables
```

## Data

Layered so that each tier removes one reviewer objection — real MC testbed traces
(channel realism), open clinical time series (decision realism), published
prompt-injection corpora (payload realism), and a calibrated synthetic corpus for
scale. Full rationale and citations: **[docs/02_datasets.md](docs/02_datasets.md)**.
Machine-readable registry: `src/agentjack/data/registry.py`.

## Reproducibility

Every run writes a JSON record with config hash, git SHA, seed and environment.
Figures and tables regenerate from those records alone. Headline numbers come
from the deterministic learned controller; the LLM is the realism demonstrator.
See **[docs/04_reproducibility.md](docs/04_reproducibility.md)**.

## Responsible disclosure

The attacks target a simulated system. No real device, protocol or deployed
clinical system is attacked, and no attack is released without the defense that
mitigates it. Payloads are reused from published defensive benchmarks.

## Build plan

15 days, one gate per day, with explicit cut lines:
**[docs/15_day_plan.md](docs/15_day_plan.md)**.

## Citing

See `CITATION.cff`. License: MIT.
