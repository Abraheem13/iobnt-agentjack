#!/usr/bin/env python
"""Figures drawn only from the supplementary result files (s1, s4)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RES, OUT, REPO = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(REPO / "src"))
from agentjack.physical.modulation import PhraseCodebook
s1 = json.load(open(RES / "s1_detection_extended.json")); s4 = json.load(open(RES / "s4_annotation_delivery.json"))

bits = [int(b) for b in s4["rows"]]
fig, ax = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True)
ax[0].bar(bits, [len(PhraseCodebook(b).persuasive_indices) for b in bits], color="C0", width=0.6)
ax[0].set_ylabel("persuasive phrases\naddressable")
ax[1].plot(bits, [100 * s4["rows"][str(b)]["delivery"] for b in bits], "o-", color="C3")
ax[1].set_ylabel("annotation\ndelivery (%)"); ax[1].set_xlabel("annotation index width $b$ (bits)"); ax[1].set_xticks(bits); ax[1].set_ylim(-5, 105)
fig.tight_layout(); fig.savefig(OUT / "fig_symbol_budget.pdf", bbox_inches="tight"); plt.close(fig)

t = s1["traffic"]; pts = [("A1_replay", "A1 replay"), ("A2_spoofing", "A2 spoofing"), ("A3_isi_exploit", "A3 ISI-exploit"), ("A4_semantic", "A4 semantic"), ("A9_zero_slot_spoofing", "A9 zero-slot spoofing")]
off = {"A1_replay": (8, -12), "A2_spoofing": (8, 2), "A3_isi_exploit": (8, 4), "A4_semantic": (-24, 11), "A9_zero_slot_spoofing": (-104, -3)}
fig, ax = plt.subplots(figsize=(4.8, 3.4))
for i, (k, lab) in enumerate(pts):
    x, y = t[k]["peak_excess"], 100 * t[k]["late_frame_valid_crc"]
    ax.plot(x, y, "o", color=f"C{i}", markersize=8); ax.annotate(lab, (x, y), textcoords="offset points", xytext=off[k], fontsize=8)
ax.set_xlabel("peak per-slot count excess"); ax.set_ylabel("forged frame forwarded\nwith valid checksum (%)"); ax.set_xlim(0, 0.7); ax.set_ylim(0, 118)
fig.tight_layout(); fig.savefig(OUT / "fig_believability.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")

# stealth window: analytic, from the channel parameters only
import numpy as np
from agentjack.channel.analytic import discrete_cir
from agentjack.channel.diffusion import ChannelParams
p = ChannelParams(); h = discrete_cir(p.isi_memory_L, p.T_s, p.D, p.r_rx, p.d_tx_rx); N = p.n_molecules
theta = 0.5 * N * h[0]; a = np.linspace(0, 1, 200); lo, hi = theta / (N * h[1]), theta / (N * h[0])
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ax.plot(a, a * N * h[0], color="C0", label=r"own slot, $\alpha N h[1]$"); ax.plot(a, a * N * h[1], color="C1", label=r"next slot, $\alpha N h[2]$")
ax.axhline(theta, color="k", ls="--", lw=1, label=r"threshold $\theta$"); ax.axvspan(lo, hi, color="C2", alpha=0.2)
ax.annotate(f"stealth window\n({lo:.2f}, {hi:.2f})", ((lo + hi) / 2, theta * 0.35), ha="center", fontsize=8, color="C2")
ax.set_xlabel(r"emission amplitude $\alpha$ (fraction of full power)"); ax.set_ylabel("expected counts"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_stealth_window.pdf", bbox_inches="tight"); print("stealth", round(lo, 3), round(hi, 3))
