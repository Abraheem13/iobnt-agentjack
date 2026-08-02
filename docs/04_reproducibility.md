# Reproducibility

## Determinism
`utils/seeding.set_all_seeds()` fixes python, numpy, torch and CUDA. Every run
record stores: config hash, git SHA, seed, package versions, GPU model.

## Run records
One JSON per cell in `results/runs/`, resumable — re-running a completed cell is
a no-op unless `--force`. Figures and tables are regenerated *only* from these
records, never from memory or a notebook.

## LLM reproducibility
The weak point. Mitigations:
- Greedy decoding (`temperature: 0.0`) for all reported runs
- Pinned model **and revision hash**
- Full prompt/response traces saved
- All headline claims backed by the learned controller, with the LLM as the
  realism demonstrator — so the paper does not rest on LLM determinism

## Artifact release checklist
- [ ] Code + `requirements.lock.txt`
- [ ] Seeded benchmark configs
- [ ] Calibration parameters from Tier A
- [ ] Per-seed result records
- [ ] Figure/table regeneration scripts
- [ ] `download_data.py` with checksums
- [ ] README quickstart verified from a clean clone
