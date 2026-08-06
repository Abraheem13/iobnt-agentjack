"""Regenerate every paper figure from committed result summaries.

Reads only results/tables/*.json, never the raw per-episode records, so this
runs anywhere the repo is checked out - no dataset, no GPU, no model weights.
That matters because figures get regenerated dozens of times during writing and
must never depend on a machine that happens to have the data mounted.

Anything a figure asserts must already be in the JSON. If a number is missing,
the figure is skipped with a message rather than being drawn from a default,
because a plot that silently invents a zero is worse than no plot.

    python scripts/make_figures.py [--outdir results/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TABLES = Path("results/tables")

ATTACK_ORDER = ["A1_replay", "A2_spoofing", "A3_isi_exploit", "A4_semantic",
                "A5_compromised_node", "A6_insider_replay", "A7_adaptive_insider"]
ATTACK_LABEL = {
    "A1_replay": "A1\nreplay",
    "A2_spoofing": "A2\nspoofing",
    "A3_isi_exploit": "A3\nISI-exploit",
    "A4_semantic": "A4\nsemantic",
    "A5_compromised_node": "A5\ninsider",
    "A6_insider_replay": "A6\ninsider replay",
    "A7_adaptive_insider": "A7\nadaptive",
}
DEFENSE_ORDER = ["none", "D1_physical", "D2_semantic", "D3_nested"]
DEFENSE_LABEL = {"none": "undefended", "D1_physical": "D1 physical",
                 "D2_semantic": "D2 semantic", "D3_nested": "D3 nested"}


def load(name: str):
    p = TABLES / name
    if not p.exists():
        print(f"  skip: {p} not found - run the matching experiment first")
        return None
    return json.loads(p.read_text())


def fig_e1_attack_efficacy(outdir: Path) -> None:
    """Attack success against an undefended agent, by adversary knowledge."""
    data = load("e1_summary.json")
    if not data:
        return
    import matplotlib.pyplot as plt

    by_attack = {}
    for key, v in data.items():
        atk, knowledge = key.split("|")
        by_attack.setdefault(atk, {})[knowledge] = v["unsafe_mean"]

    attacks = [a for a in ATTACK_ORDER if a in by_attack and a != "none"]
    present = [k for k in ["blind", "statistical", "full_cir"]
               if any(k in by_attack[a] for a in attacks)]
    if not attacks:
        print("  skip: e1 has no attack rows")
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    width = 0.8 / max(len(present), 1)
    for i, k in enumerate(present):
        vals = [by_attack[a].get(k, np.nan) for a in attacks]
        x = np.arange(len(attacks)) + i * width - 0.4 + width / 2
        ax.bar(x, vals, width, label=k.replace("_", " "))

    base = by_attack.get("none", {}).get("statistical")
    if base is not None:
        ax.axhline(base, color="0.4", ls="--", lw=1,
                   label=f"benign baseline ({base:.1%})")

    ax.set_xticks(np.arange(len(attacks)))
    ax.set_xticklabels([ATTACK_LABEL.get(a, a) for a in attacks], fontsize=8)
    ax.set_ylabel("unsafe actuation rate")
    ax.set_title("Attack efficacy against an undefended agentic gateway")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = outdir / "fig2_e1_attack_efficacy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_e2_defense_matrix(outdir: Path) -> None:
    """Unsafe rate for every attack x defense cell - the headline defense result."""
    data = load("e2_defense_comparison.json")
    if not data:
        return
    import matplotlib.pyplot as plt

    grid = {}
    for key, v in data.items():
        atk, defn = key.split("|")
        grid[(atk, defn)] = v["unsafe_mean"]

    attacks = [a for a in ATTACK_ORDER if any((a, d) in grid for d in DEFENSE_ORDER)]
    defenses = [d for d in DEFENSE_ORDER if any((a, d) in grid for a in attacks)]
    if not attacks:
        print("  skip: e2 has no attack rows")
        return

    M = np.array([[grid.get((a, d), np.nan) for d in defenses] for a in attacks])

    fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(attacks) + 1.8))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.0,
                   vmax=max(np.nanmax(M), 0.01), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            # Annotate every cell: a heatmap without numbers is unreadable in
            # print and unusable to anyone checking the claims.
            ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.5 * np.nanmax(M) else "black")
    ax.set_xticks(range(len(defenses)))
    ax.set_xticklabels([DEFENSE_LABEL.get(d, d) for d in defenses], fontsize=8)
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels([ATTACK_LABEL.get(a, a).replace("\n", " ") for a in attacks],
                       fontsize=8)
    ax.set_title("Unsafe actuation rate by attack and defense")
    fig.colorbar(im, ax=ax, label="unsafe rate", format=lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    out = outdir / "fig4_e2_defense_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_e3_ablation(outdir: Path) -> None:
    """Detection by level subset - the necessity argument for nesting."""
    data = load("e3_ablation.json")
    if not data:
        return
    import matplotlib.pyplot as plt

    subsets = list(data.keys())
    attacks = [k for k in data[subsets[0]] if k != "fpr"]
    M = np.array([[data[s].get(a, np.nan) for a in attacks] for s in subsets])

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(subsets) + 2.0))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="black" if 0.25 < M[i, j] < 0.85 else "white")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([a.replace(" ", "\n") for a in attacks], fontsize=7)
    ax.set_yticks(range(len(subsets)))
    ax.set_yticklabels([f"{s}  (FPR {data[s].get('fpr', float('nan')):.1%})"
                        for s in subsets], fontsize=8)
    ax.set_title("Detection rate by monitor level subset")
    fig.colorbar(im, ax=ax, label="detection rate", format=lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    out = outdir / "fig5_e3_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_e4_llm(outdir: Path) -> None:
    """Architecture asymmetry: semantic injection reaches the LLM, not the controller."""
    data = load("e4_llm.json")
    if not data:
        return
    import matplotlib.pyplot as plt

    rows = {k: v for k, v in data.items() if not k.startswith("_")}
    agents, cells = [], {}
    for key, v in rows.items():
        agent, atk, defn = key.split("|")
        if agent not in agents:
            agents.append(agent)
        cells[(agent, atk, defn)] = v

    attacks = [a for a in ATTACK_ORDER
               if any((ag, a, "none") in cells for ag in agents)]
    if not attacks:
        print("  skip: e4 has no attack rows")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for ax, metric, title in [
        (axes[0], "unsafe_mean", "Unsafe actuation (undefended)"),
        (axes[1], "note_followed_mean",
         "Dosed on an adversarial note\nthat telemetry did not justify"),
    ]:
        width = 0.8 / max(len(agents), 1)
        for i, ag in enumerate(agents):
            vals = [cells.get((ag, a, "none"), {}).get(metric, np.nan) for a in attacks]
            x = np.arange(len(attacks)) + i * width - 0.4 + width / 2
            ax.bar(x, vals, width, label=ag)
        ax.set_xticks(np.arange(len(attacks)))
        ax.set_xticklabels([ATTACK_LABEL.get(a, a) for a in attacks], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=8)
    meta = data.get("_meta", {})
    fig.suptitle(f"LLM vs structured-field controller  ({meta.get('backend', 'unknown')})",
                 fontsize=9)
    fig.tight_layout()
    out = outdir / "fig6_e4_llm.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"regenerating figures into {outdir}\n")

    fig_e1_attack_efficacy(outdir)
    fig_e2_defense_matrix(outdir)
    fig_e3_ablation(outdir)
    fig_e4_llm(outdir)

    print("\n  fig1 (channel validation) is written by scripts/validate_channel_physics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
