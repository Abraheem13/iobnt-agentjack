# Project charter

**Paper.** *Molecular Prompt Injection: Hijacking Agentic Bio-Cyber Interfaces and a
Nested Multi-Timescale Trust Monitor for the Internet of Bio-Nano Things.*

**Venue.** IEEE TMBMC Special Issue, *Agentic AI for Autonomous Bio-Cyber Interfaces in
Molecular Communications and IoBNT*. Submission deadline 30 December 2026.

**Build window.** 15 days of implementation.

## The claim, in one sentence
Putting an agent in the control loop of a bio-cyber gateway creates a new attack
surface — adversarial molecular emissions that hijack the agent's decisions — and
defending it requires monitoring at several timescales at once, because no single
timescale detects all attack classes.

## Two headline results
1. **The threat is real.** Four molecular prompt-injection attack classes drive an
   undefended agentic gateway to unsafe actuation at a materially non-zero rate,
   for both a learned controller and a real LLM orchestrator.
2. **The defense works, and its nesting is necessary.** A nested multi-timescale
   trust monitor cuts the unsafe-actuation rate far below single-timescale
   baselines while preserving benign task success — and the ablation shows every
   level earns its place.

## Non-goals
- No wet-lab, no hardware, no new human-subject data.
- Not a new MC detector. The detector is infrastructure, not the contribution.
- Not a claim about any deployed clinical system.

## What gets released
`IoBNT-AgentJack`: the calibrated digital-twin simulator, the agent harness, the
four-class attack suite, the nested monitor, and a seeded benchmark. The artifact
is the citation engine — it ships the same day as the preprint.

## Editorial fit (why this framing)
Three of the four guest editors work on security, authentication and trustworthy
systems; the fourth works on explainable AI and digital twins. A security +
trust + explainability paper backed by a digital twin addresses all four.
