# Datasets: what we use and why

Every dataset is registered in `src/agentjack/data/registry.py`. The paper's
data-availability statement is generated from that file, so this document and the
code cannot drift apart.

The selection is deliberately layered. A reviewer's fastest objection to a
simulation paper is *"none of this is real."* Each tier removes one version of
that objection.

## Tier A — real MC testbed traces (removes: "your channel is invented")

| Dataset | What it buys us |
|---|---|
| **Macroscale ethanol testbed** (Hofmann, Torres Gómez, Fitzek, Dressler; IEEE DataPort 2023, doi:10.21227/ytkm-xp81) | Primary calibration. Real drift-assisted traces sampled every 0.1 s with a COTS alcohol sensor. Fits CIR shape, ISI tail, sensor noise and drift. |
| **Analog network coding testbed** (IEEE DataPort) | Spoofing realism — multiple transmitters superposing in one medium is exactly the physical premise of attack A2. |
| **Proton-pumping bacteria testbed** (Grebenstein et al.; IEEE DataPort doi:10.21227/3zj6-pm05) | Bio-cyber interface realism: a measured biological transducer response plus 12 dedicated noise measurements. Its methods-and-data paper appeared in TMBMC — the venue we are targeting. |
| **Magnetic-nanoparticle duct-flow testbed** (Bartunik/Unterweger et al.; public supplement) | Flow channel plus a long binary sequence — the only Tier-A source long enough to train the learned detector on real traces. |

*Access note:* the IEEE DataPort entries need an account; access is free for IEEE
members. If any prove unavailable, the ethanol testbed alone is sufficient for
calibration and the others degrade to "related work" citations.

## Tier B — biosensor telemetry (removes: "the agent's decisions are toy")

| Dataset | What it buys us |
|---|---|
| **ShanghaiT1DM / ShanghaiT2DM** (Zhao et al., *Scientific Data* 2023; figshare doi:10.6084/m9.figshare.c.6310860) | Primary task. 12 T1DM + 100 T2DM patients, 3–14 days of CGM at 15-min resolution with meals, labs and medications. Open on figshare with no DUA — downloadable on Day 1. Makes overdose a physiologically meaningful failure. |
| **PhysioNet/CinC 2019 Sepsis** | Second task with a different attacker goal (alarm suppression, not overdose). Demonstrates the threat generalises across clinical loops. |

Splits are always **by patient**, never within a patient.

## Tier C — prompt-injection corpora (removes: "your payloads are hand-written")

| Corpus | What it buys us |
|---|---|
| **AgentDojo** (Debenedetti et al., NeurIPS 2024) — 97 tasks, 629 security cases | Payload material *and* the joint utility/security reporting convention we adopt, so our numbers read natively to the agent-security community. |
| **InjecAgent** (Zhan et al., 2024) — 1,054 cases, 17 user / 62 attacker tools | Attacker-goal taxonomy (direct harm, data stealing) mapped onto molecular actuation goals; its short "enhanced" prefixes matter under our symbol budget. |
| **BIPIA** — 250 objectives, 5 scenarios, plus defenses | Published defense baselines, so D2 is not a strawman. |
| **Agent Security Bench** — 16 attacks × 11 defenses | Coverage check that our four molecular classes map onto a recognised taxonomy. |

We reuse **payload text and taxonomy**, not their environments. Every payload
keeps a provenance tag and licence field.

**The research bit:** these payloads are text; our channel carries tens of bits.
`attacks/payload_library.compress_to_budget()` is therefore a contribution in its
own right — how much instruction-hijacking power survives a molecular bottleneck
is a question nobody has asked.

## Tier D — biological actuation dynamics (optional)
Curated **BioModels** glucose–insulin SBML models give "release drug" a
mechanistic downstream effect. Behind a feature flag; first thing to cut if the
schedule slips, with `GenericFirstOrderResponse` as fallback.

## Tier E — synthetic corpus (the primary training data)
The calibrated twin generates unlimited labelled benign/attack episodes with full
ground truth. Tier A makes it credible; Tier E makes it sufficient. Released with
fixed seeds.

## Licensing and ethics
All data is de-identified and already public. No new human-subject data is
collected, so no IRB approval is required. Attack payloads are used defensively
and released with the defense that mitigates them.
