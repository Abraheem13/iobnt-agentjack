# 15-day build plan

One gate per day. **If a day's gate fails, fix it before moving on** — a broken
foundation compounds, and the two headline results are the only things that
cannot be cut.

## Week 1 — the twin and the agent

**Day 1 — Channel core + physics gate**
Implement `channel/analytic.py`, `channel/diffusion.py`, `channel/noise.py`.
Run `make validate`.
*Gate G1:* simulated CIR within 1% of the closed-form absorbing-receiver
hitting probability. Also: download Tier B (Shanghai CGM) and Tier C corpora now
— they are open and instant.

**Day 2 — Modulation, detection, ISI**
`physical/modulation.py` (incl. the command codebook and `SYMBOL_BUDGET`),
`physical/detector.py` (threshold, decision-feedback, GRU), `channel/isi.py`.
*Gate:* BER falls monotonically with SNR; DFE beats threshold detection.

**Day 3 — Calibration against real hardware**
`channel/calibrate.py` against the macroscale ethanol testbed; flow variant
against the magnetic-nanoparticle data.
*Gate G2:* twin reproduces real traces within a stated, reported error. Produces
Figure 1 — the credibility figure.

**Day 4 — Digital twin + task 1**
`twin/digital_twin.py`, `twin/actuation.py` (the safety envelope — the primary
metric depends on this being right), `twin/biology.py`, CGM loader.
*Gate:* a random agent produces a sane, non-degenerate unsafe-action baseline.

**Day 5 — Learned controller**
Train `agent/policy_controller.py` on the benign task.
*Gate:* benign task success is high and stable across seeds. This agent carries
every headline number.

**Day 6 — LLM orchestrator**
`agent/llm_orchestrator.py`, `tools.py`, `prompts.py`. Pin model + revision.
*Gate:* the LLM reaches broadly comparable benign task success. **Timebox this
day.** If integration is still unstable at the end of Day 6, ship the
learned-controller version — still first-of-kind — and revisit on Day 13.

**Day 7 — Attacks A1 and A2**
Replay and spoofing, with the knowledge-level machinery.
*Gate:* both produce non-zero unsafe actuation against an undefended agent.

## Week 2 — attacks, defense, evidence

**Day 8 — Attacks A3 and A4**
ISI-exploiting injection (the novel mechanism) and semantic injection, including
`compress_to_budget()` and the symbol-budget sweep.
*Gate:* A3 succeeds while per-slot statistics stay unremarkable — this is what
motivates the whole nested design.

**Day 9 — HEADLINE 1**
Run `e1_attack_success` at full seed count. Freeze it.
*Gate G3:* undefended UAR lands in an informative mid-range. If it is ~0% or
~100%, retune **task difficulty**, never the attack — retuning the attack to
reach a number is the thing reviewers smell.

**Day 10 — Defense baselines**
`pla_cir.py` and `llm_guardrail.py`, thresholds calibrated on benign runs only.
*Gate:* each baseline hits the 0.05 false-positive target, and each shows its
predicted blind spot.

**Day 11 — The nested monitor**
`nested_monitor.py`, `divergence.py`, threshold calibration.
*Gate:* it beats both baselines on at least the two attack classes they each miss.

**Day 12 — HEADLINE 2 + adaptive attacker**
Run `e2_defense_comparison`; implement and run A5 against the deployed monitor.
*Gate G4:* benign task success under defense ≥ 90% of undefended. Report the
adaptive-attacker result whatever it says.

**Day 13 — Ablations + generalisation**
`e3_ablation_timescales` (the necessity claim), `e4_attacker_knowledge`,
`e5_channel_severity`, plus task 2 and the second LLM (`e6_generalisation`).
*Gate:* no proper subset of timescales covers all four attack classes.

**Day 14 — Attribution, figures, statistics**
`attribution.py`, `make_figures.py`, `make_tables.py`, `statistics.py`.
Every number in the paper regenerates from `results/runs` with one command.
*Gate:* `make figures tables` runs clean from scratch.

**Day 15 — Freeze, package, write**
Lock the environment, verify the quickstart from a clean clone, write the data
availability and limitations sections, push code + preprint together.

## Cut lines, in order
If you fall behind, drop in this order — each is chosen to leave both headline
results intact:
1. BioModels/SBML actuation (Tier D) → `GenericFirstOrderResponse`
2. Third task (`env_monitor`)
3. Second LLM (`llm_orchestrator_small`)
4. Task 2 (sepsis) — *keep if at all possible; it is the generalisation claim*
5. Adaptive attacker A5 — **never cut this**; a reviewer will ask

## Do not cut
- The physics validation figure (Day 1/3)
- The undefended headline (Day 9)
- The defense comparison (Day 12)
- The timescale ablation (Day 13) — without it the "nested" claim is decorative
