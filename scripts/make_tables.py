"""Regenerate every paper table as LaTeX, from committed result summaries.

Same contract as make_figures.py: reads results/tables/*.json only, so it runs
on any checkout without the dataset, a GPU, or model weights.

Emits booktabs-style LaTeX ready for \\input{} into the manuscript, so a number
never gets retyped by hand into the paper - transcription is where results and
manuscripts silently diverge.

    python scripts/make_tables.py [--outdir paper/tables]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TABLES = Path("results/tables")

ATTACK_ORDER = ["A1_replay", "A2_spoofing", "A3_isi_exploit", "A4_semantic",
                "A5_compromised_node", "A6_insider_replay", "A7_adaptive_insider"]
ATTACK_TEX = {
    "A1_replay": "A1 replay",
    "A2_spoofing": "A2 spoofing",
    "A3_isi_exploit": "A3 ISI-exploit",
    "A4_semantic": "A4 semantic",
    "A5_compromised_node": "A5 compromised node",
    "A6_insider_replay": "A6 insider replay",
    "A7_adaptive_insider": "A7 adaptive insider",
}
DEFENSE_ORDER = ["none", "D1_physical", "D2_semantic", "D3_nested"]
DEFENSE_TEX = {"none": "none", "D1_physical": "D1 physical",
               "D2_semantic": "D2 semantic", "D3_nested": "D3 nested"}


def load(name: str):
    p = TABLES / name
    if not p.exists():
        print(f"  skip: {p} not found - run the matching experiment first")
        return None
    return json.loads(p.read_text())


def pct(x) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def table_e1(outdir: Path) -> None:
    data = load("e1_summary.json")
    if not data:
        return
    by = {}
    for key, v in data.items():
        atk, knowledge = key.split("|")
        by.setdefault(atk, {})[knowledge] = v

    present = [k for k in ["blind", "statistical", "full_cir"]
               if any(k in by[a] for a in by)]
    attacks = [a for a in ATTACK_ORDER if a in by]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Unsafe actuation rate (\%) against an undefended agentic gateway, "
        r"by adversary knowledge. A blind adversary is not significantly worse than "
        r"benign traffic: channel knowledge is what makes these attacks work.}",
        r"\label{tab:attack-efficacy}",
        r"\begin{tabular}{l" + "r" * len(present) + "}", r"\toprule",
        "Attack & " + " & ".join(k.replace("_", " ") for k in present) + r" \\",
        r"\midrule",
    ]
    for a in attacks:
        lines.append(f"{ATTACK_TEX.get(a, a)} & "
                     + " & ".join(pct(by[a].get(k, {}).get("unsafe_mean")) for k in present)
                     + r" \\")
    if "none" in by:
        lines.append(r"\midrule")
        lines.append("benign baseline & "
                     + " & ".join(pct(by["none"].get(k, {}).get("unsafe_mean"))
                                  for k in present) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = outdir / "tab_attack_efficacy.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def table_e2(outdir: Path) -> None:
    data = load("e2_defense_comparison.json")
    if not data:
        return
    grid = {}
    for key, v in data.items():
        atk, defn = key.split("|")
        grid[(atk, defn)] = v

    attacks = [a for a in ATTACK_ORDER if any((a, d) in grid for d in DEFENSE_ORDER)]
    defenses = [d for d in DEFENSE_ORDER if any((a, d) in grid for a in attacks)]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Unsafe actuation rate (\%) by attack and defense. Lower is better. "
        r"Thresholds calibrated on benign traffic only; calibration and evaluation "
        r"seeds disjoint.}",
        r"\label{tab:defense-comparison}",
        r"\begin{tabular}{l" + "r" * len(defenses) + "}", r"\toprule",
        "Attack & " + " & ".join(DEFENSE_TEX.get(d, d) for d in defenses) + r" \\",
        r"\midrule",
    ]
    for a in attacks:
        vals = [grid.get((a, d), {}).get("unsafe_mean") for d in defenses]
        best = min((v for v in vals if v is not None), default=None)
        n_best = len([x for x in vals if x is not None and best is not None
                      and abs(x - best) < 1e-9])
        cells = []
        for v in vals:
            s = pct(v)
            # Bold the best cell so the reader sees at a glance which defense
            # actually stops this attack. Skip when every cell ties.
            if (v is not None and best is not None and abs(v - best) < 1e-9
                    and n_best < len(vals)):
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{ATTACK_TEX.get(a, a)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = outdir / "tab_defense_comparison.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def table_e3(outdir: Path) -> None:
    data = load("e3_ablation.json")
    if not data:
        return
    subsets = list(data)
    attacks = [k for k in data[subsets[0]] if k != "fpr"]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Detection rate (\%) by monitor level subset. No proper subset "
        r"covers every attack class: L0 is blind to an insider, L2 to an external "
        r"jammer, and only L1 detects a stale frame replayed by the node itself.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lr" + "r" * len(attacks) + "}", r"\toprule",
        "Levels & FPR & " + " & ".join(a.replace("_", " ") for a in attacks) + r" \\",
        r"\midrule",
    ]
    for s in subsets:
        row = [pct(data[s].get("fpr"))] + [pct(data[s].get(a)) for a in attacks]
        name = s
        if s == "L0+L1+L2":
            name = r"\textbf{L0+L1+L2}"
            row = [r"\textbf{" + x + "}" for x in row]
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = outdir / "tab_ablation.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def table_e4(outdir: Path) -> None:
    data = load("e4_llm.json")
    if not data:
        return
    meta = data.get("_meta", {})
    rows = {k: v for k, v in data.items() if not k.startswith("_")}
    cells, agents = {}, []
    for key, v in rows.items():
        agent, atk, defn = key.split("|")
        if agent not in agents:
            agents.append(agent)
        cells[(agent, atk, defn)] = v
    attacks = [a for a in ATTACK_ORDER if any((ag, a, "none") in cells for ag in agents)]

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Semantic injection reaches an LLM orchestrator but not a "
        r"structured-field controller. \emph{Note-followed} counts doses taken on a "
        r"cycle carrying an adversarial annotation where telemetry did not justify "
        f"one. Model: \\texttt{{{meta.get('backend', 'unknown')}}}.}}",
        r"\label{tab:llm}",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Agent & Attack & Unsafe & Note-followed \\", r"\midrule",
    ]
    for ag in agents:
        for a in attacks:
            c = cells.get((ag, a, "none"), {})
            lines.append(f"{ag.replace('_', ' ')} & {ATTACK_TEX.get(a, a)} & "
                         f"{pct(c.get('unsafe_mean'))} & "
                         f"{pct(c.get('note_followed_mean'))} " + r"\\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out = outdir / "tab_llm.tex"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="paper/tables")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"regenerating LaTeX tables into {outdir}\n")

    table_e1(outdir)
    table_e2(outdir)
    table_e3(outdir)
    table_e4(outdir)

    print("\n  \\input{tables/<name>} these from paper/main.tex - never retype a number")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
