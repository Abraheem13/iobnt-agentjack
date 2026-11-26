"""Regenerate every result file the manuscript reads, in one go.

    python scripts/reproduce_paper.py                 # everything
    python scripts/reproduce_paper.py --skip-llm      # everything except the language-model protocol
    python scripts/reproduce_paper.py --only-llm      # only the language-model protocol

Stage 1 (CPU or GPU, no language model): channel validation, efficacy (e1),
defence (e2), ablation (e3), supplementary analyses (s1-s5).
Stage 2 (GPU): language-model protocol (e4), then figures and tables.

Each stage writes to results/tables and results/figures, and a log per step to
results/tables/logs. With --push, each stage is committed and pushed to the
branch given by --branch, so the manuscript can be updated from stage 1 while
stage 2 is still running.

The clinical cohort must be on disk (data/raw/shanghai_cgm). Without it the
scripts silently fall back to a synthetic cohort, which is not what the paper
reports, so this driver refuses to start unless --allow-synthetic is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLES = REPO / "results" / "tables"
LOGS = TABLES / "logs"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REV = "a09a35458c702b33eeacc393d103063234e8bc28"


def run(name: str, cmd: list[str], status: dict) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{name}.log"
    print(f"\n=== {name}: {' '.join(cmd)}\n    log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
        code = proc.wait()
    status[name] = {"exit_code": code, "minutes": round((time.time() - t0) / 60, 1),
                    "command": " ".join(cmd)}
    (TABLES / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"=== {name}: exit {code} after {status[name]['minutes']} min", flush=True)
    return code


def push(branch: str, message: str) -> None:
    def git(*a: str) -> int:
        return subprocess.call(["git", *a], cwd=REPO)

    git("checkout", "-B", branch)
    git("add", "-f", "results/tables", "results/figures")
    if git("diff", "--cached", "--quiet") != 0:
        git("commit", "-m", message)
    if git("push", "-u", "origin", branch, "--force-with-lease") != 0:
        print("!! git push failed - push the branch by hand:  git push -u origin " + branch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--only-llm", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--push", action="store_true", help="commit and push results after each stage")
    ap.add_argument("--branch", default="results-rerun")
    ap.add_argument("--e1-episodes", type=int, default=8, help="episodes per seed in the efficacy protocol")
    ap.add_argument("--llm-seeds", type=int, default=2)
    ap.add_argument("--llm-episodes", type=int, default=4,
                    help="episodes per seed; 4 or more covers all four held-out patients")
    ap.add_argument("--llm-quant", default="4bit", choices=["none", "8bit", "4bit"],
                    help="4bit fits a 16 GB GPU; use none on a 24 GB+ GPU")
    ap.add_argument("--llm-prompt", default="both", choices=["v1_plain", "v1_hardened", "both"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    py = sys.executable
    status: dict = {}

    sys.path.insert(0, str(REPO / "src"))
    from agentjack.data.loaders.cgm import load_cgm_or_synthetic

    import os
    os.chdir(REPO)
    cgm, is_real = load_cgm_or_synthetic(seed=0)
    print(f"cohort: {'CLINICAL (ShanghaiT1DM)' if is_real else 'SYNTHETIC FALLBACK'}, "
          f"{cgm.patient_id.nunique()} patients")
    if not is_real and not args.allow_synthetic:
        print("\nThe clinical cohort is not in data/raw/shanghai_cgm. Download the ShanghaiT1DM\n"
              "files from https://doi.org/10.6084/m9.figshare.c.6310860 and unzip them there\n"
              "(the .xls/.xlsx patient files, in a folder whose path contains 'T1DM').\n"
              "Refusing to run on the synthetic fallback; pass --allow-synthetic to override.")
        return 2

    if not args.only_llm:
        # Stale per-run records from earlier settings would be averaged into e1.
        runs = REPO / "results" / "runs" / "e1"
        if runs.exists():
            shutil.move(str(runs), str(runs.with_name(f"e1_old_{int(time.time())}")))
        run("00_channel_validation", [py, "scripts/validate_channel_physics.py", "--no-figure"], status)
        run("01_e1_efficacy", [py, "scripts/run_e1.py", "--seeds", "10",
                               "--episodes", str(args.e1_episodes), "--force"], status)
        run("02_e2_defence", [py, "scripts/run_e2.py", "--seeds", "10", "--episodes", "8"], status)
        run("03_e3_ablation", [py, "scripts/validate_defense.py"], status)
        run("04_supplementary", [py, "scripts/run_supplementary.py", "s1", "s2", "s3", "s4", "s5"], status)
        run("05_supp_figures", [py, "scripts/make_supplementary_figures.py", "results/tables",
                                "results/figures", "."], status)
        if args.push:
            push(args.branch, "Results: stage 1 (e1, e2, e3, s1-s5) rerun with monitor reset")

    if not args.skip_llm:
        ckpt = REPO / "results" / "runs" / "e4_checkpoint.json"
        marker = REPO / "results" / "runs" / "e4_checkpoint.settings.json"
        settings = {"seeds": args.llm_seeds, "episodes": args.llm_episodes, "quant": args.llm_quant,
                    "prompt": args.llm_prompt, "monitor_reset": True}
        # Resume only a checkpoint written with the same settings by this driver.
        if ckpt.exists() and not (marker.exists() and json.loads(marker.read_text()) == settings):
            shutil.move(str(ckpt), str(ckpt.with_name(f"e4_checkpoint_old_{int(time.time())}.json")))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(settings))
        run("06_e4_llm", [py, "scripts/run_e4_llm.py", "--model", QWEN, "--revision", QWEN_REV,
                          "--device", args.device, "--prompt", args.llm_prompt,
                          "--seeds", str(args.llm_seeds), "--episodes", str(args.llm_episodes),
                          "--max-new-tokens", "16", "--quant", args.llm_quant, "--resume"], status)

    run("07_figures", [py, "scripts/make_figures.py"], status)
    run("08_tables", [py, "scripts/make_tables.py"], status)
    if args.push:
        push(args.branch, "Results: full rerun (e1-e4, s1-s5) with monitor reset")

    print("\nSummary")
    for k, v in status.items():
        print(f"  {k:<24} exit {v['exit_code']}  {v['minutes']} min")
    print("\nA non-zero exit on e1/e2/e3 means one of that script's own PASS/FAIL gates failed;\n"
          "the result file is still written. Read the log before using the numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
