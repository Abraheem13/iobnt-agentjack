#!/usr/bin/env python
"""All data figures of the manuscript, in one style, from the result files only.

    python scripts/make_paper_figures.py [results/tables] [outdir]

Needs no PyTorch. The channel-validation panel reruns the particle simulation
of scripts/validate_channel_physics.py (same parameters, same seed).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "tables"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# One style for every figure: the manuscript's text face, small sizes that stay
# legible at column width, one blue for data, grey for reference lines.
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["LM Roman 10", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
BLUE, MID, LIGHT, GREY, ORANGE = "#1f4e79", "#5b8db8", "#b7cde3", "#6b6b6b", "#c8641e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#dbe7f2", "#8fb3d4", "#3f76a8", "#1f4e79"])
W = 5.0  # inches; figures are included at about this width


def load(name):
    return json.load(open(RES / name))


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def heat(ax, M, fmt="{:.1f}"):
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 55 else "black")
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# ---------------------------------------------------------------- channel validation
def fig_channel():
    from agentjack.channel.analytic import asymptotic_absorption, cumulative_hitting_probability
    from agentjack.channel.diffusion import ChannelParams
    from validate_channel_physics import check_monte_carlo

    p = ChannelParams()
    _, msg, analytic, mc, _ = check_monte_carlo(p, 40_000, 400)
    print("channel:", msg)
    k = np.arange(1, len(analytic) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(W + 0.6, 2.3))
    ml, sl, _ = ax[0].stem(k, analytic, linefmt="-", markerfmt="o", basefmt=" ", label="analytic")
    plt.setp(ml, color=BLUE, markersize=4)
    plt.setp(sl, color=BLUE, linewidth=1.0)
    ax[0].plot(k, mc, "x", color=ORANGE, markersize=6, markeredgewidth=1.1,
               label="particle simulation")
    ax[0].set_xlabel("slot $k$")
    ax[0].set_ylabel("$h[k]$")
    ax[0].set_xticks(k)
    ax[0].set_ylim(0, 1.45 * max(analytic.max(), mc.max()))
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set_title("(a)", loc="left")
    t = np.linspace(1e-4, p.isi_memory_L * p.T_s, 2000)
    ax[1].plot(t, cumulative_hitting_probability(t, p.D, p.r_rx, p.d_tx_rx), color=BLUE, label="$F(t)$")
    ax[1].axhline(asymptotic_absorption(p.r_rx, p.d_tx_rx), color=GREY, ls="--", lw=0.9,
                  label=r"ceiling $r_{\mathrm{rx}}/d$")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("cumulative absorption")
    ax[1].set_ylim(0, 0.29)
    ax[1].legend(frameon=False, loc="lower right")
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_channel_validation.pdf")


# ---------------------------------------------------------------- efficacy (e1)
def fig_efficacy():
    d = load("e1_summary.json")
    runs = load("e1_per_run.json")
    atk = [("A1_replay", "A1 replay"), ("A2_spoofing", "A2 spoofing"),
           ("A3_isi_exploit", "A3 ISI-exploiting"), ("A4_semantic", "A4 semantic")]
    kn = [("blind", "blind", LIGHT), ("statistical", "statistical", MID), ("full_cir", "full CIR", BLUE)]
    fig, ax = plt.subplots(figsize=(W, 2.5))
    w = 0.26
    for i, (k, lab, col) in enumerate(kn):
        x = np.arange(len(atk)) + (i - 1) * w
        v = [100 * d[f"{a}|{k}"]["unsafe_mean"] for a, _ in atk]
        ax.bar(x, v, w * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.4)
        for xi, (a, _), vi in zip(x, atk, v):
            seeds = [100 * r["summary"]["unsafe_rate"] for r in runs if r["attack"] == a and r["knowledge"] == k]
            ax.plot([xi] * len(seeds), seeds, ".", color="black", markersize=2.2, alpha=0.55, zorder=3)
            ax.text(xi, max(seeds + [vi]) + 1.5, f"{vi:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(atk)))
    ax.set_xticklabels([l for _, l in atk])
    ax.set_ylabel("unsafe actuation (%)")
    ax.set_ylim(0, 50)
    ax.legend(frameon=False, title="adversary knowledge", title_fontsize=8, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig2_e1_attack_efficacy.pdf")


# ---------------------------------------------------------------- defence matrix (e2)
def fig_defence():
    d = load("e2_defense_comparison.json")
    A = [("A1_replay", "A1 replay"), ("A2_spoofing", "A2 spoofing"), ("A3_isi_exploit", "A3 ISI-exploiting"),
         ("A4_semantic", "A4 semantic"), ("A5_compromised_node", "A5 compromised node"),
         ("A6_insider_replay", "A6 insider replay"), ("A7_adaptive_insider", "A7 adaptive insider")]
    D = [("none", "undefended"), ("D1_physical", "D1\nphysical"), ("D2_semantic", "D2\nsemantic"), ("D3_nested", "D3\nnested")]
    M = np.array([[100 * d[f"{a}|{x}"]["unsafe_mean"] for x, _ in D] for a, _ in A])
    fig, ax = plt.subplots(figsize=(W - 0.4, 3.0))
    im = heat(ax, M)
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels([l for _, l in D])
    ax.set_yticks(range(len(A)))
    ax.set_yticklabels([l for _, l in A])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("unsafe actuation (%)")
    cb.outline.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, "fig4_e2_defense_matrix.pdf")


# ---------------------------------------------------------------- ablation (e3)
def fig_ablation():
    d = load("e3_ablation.json")
    rows = list(d)
    cols = [("A1 replay", "A1\nreplay"), ("A2 spoofing", "A2\nspoofing"), ("A3 ISI-exploit", "A3\nISI-\nexploiting"),
            ("A4 semantic", "A4\nsemantic"), ("A5 compromised node", "A5\ncompromised\nnode"),
            ("A6 insider replay", "A6\ninsider\nreplay")]
    M = np.array([[100 * d[r][c] for c, _ in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(W + 0.4, 3.1))
    im = heat(ax, M)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([l for _, l in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["$" + "{+}".join(f"L_{p[1]}" for p in r.split("+")) + "$" for r in rows])
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("cycles flagged (%)")
    cb.outline.set_linewidth(0.4)
    fig.tight_layout()
    save(fig, "fig5_e3_ablation.pdf")


# ---------------------------------------------------------------- believability (s1)
def fig_believability():
    t = load("s1_detection_extended.json")["traffic"]
    pts = [("A1_replay", "A1", (-7, -11)), ("A2_spoofing", "A2", (6, 3)), ("A3_isi_exploit", "A3", (7, -3)),
           ("A4_semantic", "A4", (-13, 5)), ("A9_zero_slot_spoofing", "A9", (-13, 5))]
    fig, ax = plt.subplots(figsize=(W - 1.2, 2.6))
    for k, lab, off in pts:
        x, y = t[k]["peak_excess"], 100 * t[k]["late_frame_valid_crc"]
        ax.plot(x, y, "o", color=BLUE, markersize=5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xlabel("peak per-slot count excess\n(fraction of peak legitimate count)")
    ax.set_ylabel("forged frame forwarded\nwith valid checksum (%)")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 108)
    fig.tight_layout()
    save(fig, "fig_believability.pdf")


# ---------------------------------------------------------------- symbol budget (s4)
def fig_budget():
    from agentjack.physical.modulation import PhraseCodebook

    s4 = load("s4_annotation_delivery.json")["rows"]
    bits = [int(b) for b in s4]
    n = [len(PhraseCodebook(b).persuasive_indices) for b in bits]
    dl = [100 * s4[str(b)]["delivery"] for b in bits]
    fig, ax = plt.subplots(2, 1, figsize=(W - 1.4, 3.3), sharex=True)
    ax[0].bar(bits, n, color=BLUE, width=0.55)
    for b, v in zip(bits, n):
        ax[0].text(b, v + 0.15, str(v), ha="center", va="bottom", fontsize=8)
    ax[0].set_ylabel("persuasive phrases\naddressable")
    ax[0].set_ylim(0, max(n) * 1.25 + 0.5)
    ax[0].set_title("(a)", loc="left")
    ax[1].plot(bits, dl, "o-", color=BLUE, markersize=4)
    for b, v in zip(bits, dl):
        ax[1].text(b, v + 6 if v < 50 else v - 14, f"{v:.1f}", ha="center", fontsize=8)
    ax[1].set_ylabel("annotation\ndelivery (%)")
    ax[1].set_xlabel("annotation index width $b$ (bits)")
    ax[1].set_xticks(bits)
    ax[1].set_ylim(-5, 112)
    ax[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, "fig_symbol_budget.pdf")


# ---------------------------------------------------------------- stealth window (analytic)
def fig_stealth():
    from agentjack.channel.analytic import discrete_cir
    from agentjack.channel.diffusion import ChannelParams

    p = ChannelParams()
    h = discrete_cir(p.isi_memory_L, p.T_s, p.D, p.r_rx, p.d_tx_rx)
    N = p.n_molecules
    theta = 0.5 * N * h[0]
    a = np.linspace(0, 1, 200)
    lo, hi = theta / (N * h[1]), theta / (N * h[0])
    print("stealth window", round(lo, 3), round(hi, 3))
    fig, ax = plt.subplots(figsize=(W - 1.2, 2.6))
    ax.axvspan(lo, hi, color=LIGHT, alpha=0.6, linewidth=0, label=f"stealth window $({lo:.2f},\\,{hi:.2f})$")
    ax.plot(a, a * N * h[0], color=BLUE, label=r"own slot, $\alpha N h[1]$")
    ax.plot(a, a * N * h[1], color=ORANGE, ls="-.", label=r"next slot, $\alpha N h[2]$")
    ax.axhline(theta, color=GREY, ls="--", lw=0.9, label=r"threshold $\theta$")
    ax.set_xlabel(r"emission amplitude $\alpha$ (fraction of full power)")
    ax.set_ylabel("expected count per slot (molecules)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_stealth_window.pdf")


if __name__ == "__main__":
    fig_efficacy(); fig_defence(); fig_ablation(); fig_believability(); fig_budget(); fig_stealth(); fig_channel()
