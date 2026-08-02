# Experimental protocol

Fixed before results are seen. Deviations get recorded in this file with a date.

## Primary metric
**Unsafe-actuation rate (UAR)** — the fraction of episodes in which the agent
executes an action violating the task's safety envelope. Chosen because it is the
thing that actually matters: detection is a means, not the end.

## Secondary metrics
- Attack success rate (attacker's specific goal achieved)
- Benign task success (utility must survive the defense)
- Detection AUROC, detection latency (in symbol slots)
- False-positive rate on benign runs (must hold at the 0.05 target)
- Symbol error rate / BER (channel sanity)
- Attribution accuracy (did the monitor blame the right timescale?)

## Design
Full factorial over task × channel × agent × attack × defense, 10 seeds for
headline experiments and 5 for sweeps. Seeds are shared across arms so every
comparison is **paired**.

## Statistics
Mirrors the TierFed protocol:
- Paired tests across shared seeds
- Holm–Bonferroni correction over the whole comparison family
- BCa bootstrap confidence intervals (10,000 resamples)
- Hedges' *g* with small-sample correction
- Parity is never inferred from a non-significant difference — use an explicit
  non-inferiority test with a pre-registered margin

## Calibration discipline
Defense thresholds are fitted on **benign runs only**, to a fixed false-positive
target. Fitting thresholds on attack data would be the single most damaging
methodological error available to us.

## Gates
| Gate | When | Criterion | If it fails |
|---|---|---|---|
| G1 physics | Day 1 | Simulated CIR within 1% of closed form | Stop; fix the channel |
| G2 calibration | Day 3 | Twin reproduces testbed traces within stated error | Report honestly, keep synthetic-only as primary |
| G3 informativeness | Day 9 | Undefended UAR in an informative mid-range | Retune task difficulty, not the attack |
| G4 utility | Day 12 | Benign task success under defense ≥ 90% of undefended | Loosen thresholds and report the trade-off |

## Pre-registered honesty commitments
- Report every configuration run, including the ones that fail.
- Report where the defense does **not** help.
- Report the LLM's variance across decoding seeds.
- Never quietly drop a seed.
