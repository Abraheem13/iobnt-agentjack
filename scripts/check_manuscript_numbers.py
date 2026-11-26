#!/usr/bin/env python
"""Check every result-table cell of main.tex against the result files."""
import json, re, sys
from pathlib import Path
TEX, REPO, SUPP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
s = TEX.read_text(); T = REPO / "results" / "tables"
e1 = json.load(open(T / "e1_summary.json")); e2 = json.load(open(T / "e2_defense_comparison.json"))
e3 = json.load(open(T / "e3_ablation.json")); e4 = json.load(open(T / "e4_llm.json"))
s1 = json.load(open(SUPP / "s1_detection_extended.json")); s2 = json.load(open(SUPP / "s2_rule_based_closed_loop.json"))
def table(label):
    i = s.index("\\label{%s}" % label); a = s.index("\\midrule", i); b = s.index("\\bottomrule", a)
    rows = []
    for line in s[a:b].split("\\\\"):
        line = re.sub(r"\\(midrule|cmidrule\([^)]*\)\{[^}]*\})", "", line).strip()
        if "&" in line: rows.append([re.sub(r"\\textbf\{([^}]*)\}|\\multicolumn\{\d\}\{c\}\{([^}]*)\}", lambda m: m.group(1) or m.group(2), c).strip() for c in line.split("&")])
    return rows
bad = n = 0
def chk(what, shown, value):
    global bad, n; n += 1
    if f"{100*value:.1f}" != shown: bad += 1; print("MISMATCH", what, "tex", shown, "file", f"{100*value:.1f}")
print("keys e1:", list(e1)[:3], "| e2:", list(e2)[:2], "| e3:", list(e3)[:2], "| e4:", [k for k in e4][:3])
ATT = ["A1_replay", "A2_spoofing", "A3_isi_exploit", "A4_semantic", "A5_compromised_node", "A6_insider_replay", "A7_adaptive_insider", "A8_nonce_refreshing_insider", "A9_zero_slot_spoofing"]
SHORT = {a.split("_")[0]: a for a in ATT}
# Table: attack efficacy (e1)
for r in table("tab:attack-efficacy"):
    if r[0] in SHORT:
        for col, k in zip(r[1:], ["blind", "statistical", "full_cir"]): chk(f"e1 {r[0]} {k}", col, e1[f"{SHORT[r[0]]}|{k}"]["unsafe_mean"])
# Table: defence comparison (e2)
for r in table("tab:defence-comparison"):
    if r[0] in SHORT:
        for col, d in zip(r[1:], ["none", "D1_physical", "D2_semantic", "D3_nested"]): chk(f"e2 {r[0]} {d}", col, e2[f"{SHORT[r[0]]}|{d}"]["unsafe_mean"])
# Table: ablation (e3)
E3 = ["fpr", "A1 replay", "A2 spoofing", "A3 ISI-exploit", "A4 semantic", "A5 compromised node", "A6 insider replay"]
for r in table("tab:ablation"):
    lev = re.sub(r"\\bm|[$_{}]", "", r[0])
    for col, k in zip(r[1:], E3): chk(f"e3 {lev} {k}", col, e3[lev][k])
# Table: language model (e4)
AG = {"controller": "policy_gru", "LLM, plain prompt": "llm[v1_plain]", "LLM, hardened": "llm[v1_hardened]"}
for r in table("tab:llm"):
    a = "A4_semantic" if r[1].startswith("A4") else "A5_compromised_node"
    for col, d in zip(r[2:], ["none", "D1_physical", "D2_semantic", "D3_nested"]): chk(f"e4 {r[0]} {a} {d}", col, e4[f"{AG[r[0]]}|{a}|{d}"]["note_followed_mean"])
# Supplementary tables
LV = {"$L_0$": "L0", "$L_1$": "L1", "$L_2$": "L2", "$L_0{+}L_1{+}L_2$": "L0+L1+L2"}
for r in table("tab:supp-detection"):
    for col, a in zip(r[1:], ["none"] + ATT): chk(f"s1 {r[0]} {a}", col, s1["detection"][LV[r[0]]][a]["veto_rate"])
for r in table("tab:supp-closed-loop"):
    if r[0] not in SHORT and r[0] != "benign": continue
    a = "none" if r[0] == "benign" else SHORT[r[0]]; D = ["none", "D1_physical", "D2_semantic", "D3_nested"]
    for col, d in zip(r[1:5], D): chk(f"s2 expert {a} {d}", col, s2["cells"][f"expert|{a}|{d}"]["below_3p9_mean"])
    for col, d in zip(r[5:9], D): chk(f"s2 follower {a} {d}", col, s2["cells"][f"follow_request|{a}|{d}"]["unsafe_mean"])
    chk(f"s2 veto {a}", r[9], s2["cells"][f"expert|{a}|D3_nested"]["veto_mean"])
print(f"{n} table cells checked, {bad} mismatches")
# Selected prose numbers from the repository result files
P = {"98.7": e2["none|none"]["tir_mean"], "32.0": e2["A1_replay|none"]["tir_mean"], "27.3": e2["A2_spoofing|none"]["tir_mean"], "67.9": e2["A3_isi_exploit|none"]["tir_mean"],
     "38.0": e2["A6_insider_replay|none"]["tir_mean"], "73.5": e2["A6_insider_replay|D3_nested"]["veto_mean"], "29.9": e2["A7_adaptive_insider|D3_nested"]["veto_mean"],
     "24.5": e2["A7_adaptive_insider|D2_semantic"]["veto_mean"], "7.4": e2["A7_adaptive_insider|D1_physical"]["veto_mean"], "7.1": e2["none|D1_physical"]["veto_mean"],
     "54.1": s2["cells"]["expert|A1_replay|D3_nested"]["veto_mean"], "45.8": e4["policy_gru|A2_spoofing|none"]["unsafe_mean"], "90.3": e4["llm[v1_plain]|none|none"]["tir_mean"],
     "83.5": e4["llm[v1_hardened]|none|none"]["tir_mean"], "97.9": e4["policy_gru|none|none"]["tir_mean"], "54.0": min(e2[f"{SHORT[a]}|D3_nested"]["veto_mean"] for a in ["A1","A2","A3","A4","A5","A6"]),
     "82.1": max(e2[f"{SHORT[a]}|D3_nested"]["veto_mean"] for a in ["A1","A2","A3","A4","A5","A6"]), "91.6": min(e2[f"{SHORT[a]}|D3_nested"]["tir_mean"] for a in ["A1","A2","A3","A4","A5","A6"]),
     "97.1": max(e2[f"{SHORT[a]}|D3_nested"]["tir_mean"] for a in ["A1","A2","A3","A4","A5","A6"]), "68.0": e2["A1_replay|none"]["below_3p9_mean"], "60.8": e2["A6_insider_replay|D1_physical"]["below_3p9_mean"],
     "98.1": min(e2[f"none|{d}"]["tir_mean"] for d in ["D1_physical","D2_semantic","D3_nested"])}
pb = 0
for shown, v in P.items():
    ok = f"{100*v:.1f}" == shown and (f"${shown}\\%$" in s or f"${shown}$" in s)
    pb += not ok
    if not ok: print("PROSE MISMATCH", shown, f"{100*v:.1f}", "present" if shown in s else "absent")
print(f"{len(P)} prose values checked, {pb} mismatches")
sys.exit(1 if bad or pb else 0)
