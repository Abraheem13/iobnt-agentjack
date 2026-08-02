# Local development on macOS

Days 1-5 run entirely on a MacBook. Day 6 (the LLM orchestrator) is where you
move to the GPU box. The learned-controller arms of every experiment stay
runnable locally throughout, which is why the headline numbers are pinned to
that agent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"          # NOT ".[dev,llm]" on the Mac
python scripts/doctor.py
```

`vllm` is already gated to Linux in `pyproject.toml`, so the `llm` extra will not
break the install here - it just pulls weight you do not need until Day 6.

## Device policy

Three rules, in order of how badly breaking them hurts:

1. **Channel physics stays NumPy float64 on CPU.** MPS has no float64. Casting
   the CIR to float32 to put it on the accelerator would quietly degrade the
   Gate-1 validation against the closed-form hitting probability - you would be
   comparing a damaged simulation against exact analysis and calling the gap
   physics.
2. **Never extend one results table across devices.** MPS and CUDA are not
   bit-identical. Pick one device per experiment family and finish it there.
   Every run record stores the device, so this is checkable after the fact.
3. **Only detector and controller tensors move to the accelerator.**

Resolve devices through `agentjack.utils.seeding.resolve_device`, never by
hard-coding a string.

```python
from agentjack.utils.seeding import resolve_device, set_all_seeds
set_all_seeds(cfg.seed)
device = resolve_device(cfg.device)   # "auto" -> cuda | mps | cpu
```

## What runs where

| Days | Work | Mac | GPU box |
|---|---|---|---|
| 1 | Channel core, analytic validation | yes | not needed |
| 2 | Modulation, detectors, ISI | yes (MPS or CPU) | faster |
| 3 | Testbed calibration | yes | not needed |
| 4 | Digital twin, actuation, CGM task | yes | not needed |
| 5 | Learned controller training | yes - small GRU, tens of minutes | faster |
| 6 | LLM orchestrator | quantised 7B possible with 32GB+, but slow | **yes** |
| 7-8 | Attacks A1-A4 | controller arms yes | LLM arms yes |
| 9-13 | Full experiment grids | no - throughput | **yes** |
| 14-15 | Figures, tables, writing | yes | not needed |

The Day 6 break point is throughput, not capability: E2 is
agent x attack x defense x seeds x episodes, which is thousands of generations.

## Moving to the GPU box

```bash
git push
# on the GPU machine:
git clone <repo> && cd iobnt-agentjack
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
python scripts/doctor.py            # should report device: cuda
```

Sync `data/` and `results/runs/` with rsync - both are git-ignored by design.

## Apple Silicon gotchas

- `PYTORCH_ENABLE_MPS_FALLBACK=1` lets unsupported ops fall back to CPU instead
  of raising. Useful, but it hides silent slowness - do not leave it on for
  timing measurements.
- `torch.use_deterministic_algorithms` is largely a no-op on MPS. Determinism
  claims in the paper rest on the CPU/CUDA paths.
- Activate the venv in every new shell, or `make` targets will use system python.
