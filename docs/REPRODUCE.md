# Reproducing the paper's results

One command regenerates every result file the manuscript reads
(`results/tables/*.json`, `results/figures/*.pdf`).

## 1. Environment (NVIDIA RTX 50-series, e.g. RTX 5080 16 GB)

RTX 50-series cards need a PyTorch build for CUDA 12.8 (`cu128`, PyTorch 2.7 or newer).

Linux / WSL:

```bash
git clone https://github.com/Abraheem13/iobnt-agentjack.git
cd iobnt-agentjack
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install "transformers>=4.44" "accelerate>=0.33" sentencepiece "bitsandbytes>=0.44" openpyxl xlrd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Windows (PowerShell): the same, with `py -3 -m venv .venv; .venv\Scripts\Activate.ps1`.

Do not install the `llm` extra (`pip install -e ".[llm]"`): it pulls in vLLM, which
can replace the CUDA 12.8 PyTorch build.

## 2. Clinical cohort

Download the ShanghaiT1DM files from <https://doi.org/10.6084/m9.figshare.c.6310860>
and unzip them into `data/raw/shanghai_cgm/` so that the patient workbooks
(`*.xls`, `*.xlsx`) sit in a folder whose path contains `T1DM`. Check:

```bash
python -c "import sys; sys.path.insert(0,'src'); from agentjack.data.loaders.cgm import load_cgm_or_synthetic as f; d,r=f(); print('clinical' if r else 'SYNTHETIC', d.patient_id.nunique(), 'patients')"
```

It must print `clinical 12 patients`.

## 3. Run

```bash
python scripts/reproduce_paper.py --push --branch results-rerun
```

- Stage 1 (efficacy e1, defence e2, ablation e3, supplementary s1-s5) needs no
  language model. With `--push` it is committed and pushed as soon as it ends.
- Stage 2 is the language-model protocol (e4): Qwen2.5-7B-Instruct at revision
  `a09a354`, greedy decoding, 16 new tokens. Defaults: 2 seed blocks x 4 episodes
  (8 episodes per condition, all four held-out patients), both prompts, 4-bit
  quantisation so the model fits in 16 GB. It checkpoints after every condition;
  rerunning the same command resumes.
- On a 24 GB or larger GPU use `--llm-quant none` to run the unquantised model.
- `--skip-llm` and `--only-llm` run one stage only.

Logs are written to `results/tables/logs/`, and `results/tables/run_status.json`
records each step's command, exit code and duration.

## What changed relative to the first release

- `run_e2.py` and `run_e4_llm.py` now reset the monitor at the start of every
  evaluation episode. Before, nonce history leaked across episodes and the first
  frame of each episode was vetoed.
- Provenance is recorded: `e1_meta.json`, `e2_meta.json`, `e3_meta.json` and the
  `_meta` block of `e4_llm.json` hold the cohort source (`cgm_is_real`), seeds,
  episodes and the command line.
- Per-run records are saved: `e1_per_run.json`, `e2_per_episode.json`,
  `e4_per_episode.json`.
- Time below 3.9 mmol/L is reported per condition (`below_3p9_mean`).
- The language-model protocol reports `offpolicy_dose_mean`, the rate at which
  the agent doses where the expert would not, on every cycle. On benign traffic
  this is the note-absent baseline for the note-following rate.
- A quantised model is a different experimental condition from the unquantised
  one; `_meta.quantisation` records which was used and the paper must say so.
