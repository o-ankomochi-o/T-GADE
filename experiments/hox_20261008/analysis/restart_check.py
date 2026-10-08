# Post-hoc (not preregistered) check: k restarts x 200 total calls, one run selected by TRAINING value, versus a single
# 800-call run. Both EoH and T-GADE receive the same restart option. Descriptive only (subsets overlap).
# Usage: python restart_check.py
import sys
from pathlib import Path
D = Path(__file__).resolve().parents[1]  # experiments/hox_20261008
sys.path.insert(0, str(D / "scorers"))
import hoxN_eval_sealed as ev  # wilcoxon_exact, select
import json, itertools, statistics
from pathlib import Path
P = D; med = statistics.median; mean = statistics.mean
R7 = json.load(open(D / "derived/readout_20261007.json", encoding="utf-8"))["hox9"]
G7 = json.load(open(D / "derived/grid7_readout.json", encoding="utf-8"))["continuous"]


def sealed(paths):
    by = {}
    for p in paths:
        for r in json.load(open(p, encoding="utf-8"))["runs"]: by[(r["arm"], r["seed"])] = r["at"]
    return by


S = D / "readouts"
S9 = sealed([S / "hox9_sealed_c100.json"]); SC = sealed([S / "hox2c_sealed_c100.json", S / "hox2ef_sealed_c100.json", S / "hox11c_sealed_c100.json"])


def val(at, b):
    x = at.get(b) or at.get("end"); return 100 * x["value"]


def restart(train, seal, seeds, b, k):
    sel, orc = [], []
    for sub in itertools.combinations(seeds, k):
        best = min(sub, key=lambda s: train[s][b]); sel.append(seal[s := best][b])
        orc.append(min(seal[s][b] for s in sub))
    return sel, orc


def row(name, single_vals, sel, orc):
    return (f"| {name} | {med(single_vals):.3f} / {mean(single_vals):.3f} / {max(single_vals):.3f} | {med(sel):.3f} / {mean(sel):.3f} / {max(sel):.3f} | {med(orc):.3f} |")


print("### hox9 (seeds 86001-86010), sealed c100 %, median / mean / worst")
print("| arm | single run, 800 calls (EoH: 800; host arms: endpoint <= 800) | 4 runs x 200 calls, selected by training value | (reference) selected by sealed value (not a valid procedure) |\n|---|---|---|---|")
for arm, lab in (("eoh800", "A: EoH"), ("h_improve", "B: steady-state Fermi-type T=0 + host"), ("h_improve_T0.003", "C: T=0.003 + host")):
    seeds = list(range(86001, 86011))
    tr = {s: {b: R7[f"{arm}_{s}"]["train"][b] for b in ("200", "400", "800", "end")} for s in seeds}
    se = {s: {b: val(S9[(arm, s)], b) for b in ("200", "400", "800", "end")} for s in seeds}
    single = [se[s]["800"] for s in seeds]
    sel, orc = restart(tr, se, seeds, "200", 4); print(row(lab + ", 200x4", single, sel, orc))
    sel2, orc2 = restart(tr, se, seeds, "400", 2); print(row(lab + ", 400x2", single, sel2, orc2))
    print(f"|   (single-run median: 200={med(se[s]['200'] for s in seeds):.3f}, 400={med(se[s]['400'] for s in seeds):.3f}) | | | |")
# how often does a T-GADE restart bundle beat EoH's single 800? compare every 4-subset of B/C against every EoH seed at 800 (descriptive)
eoh800 = [val(S9[("eoh800", s)], "800") for s in range(86001, 86011)]
for arm, lab in (("h_improve", "B"), ("h_improve_T0.003", "C")):
    seeds = list(range(86001, 86011)); tr = {s: {"200": R7[f"{arm}_{s}"]["train"]["200"]} for s in seeds}; se = {s: {"200": val(S9[(arm, s)], "200")} for s in seeds}
    sel, _ = restart(tr, se, seeds, "200", 4); wins = sum(x < y for x in sel for y in eoh800); ties = sum(x == y for x in sel for y in eoh800)
    print(f"  {lab} 200×4 (training-selected) < EoH 800: {wins}/{len(sel)*len(eoh800)} pairs ({100*wins/(len(sel)*len(eoh800)):.0f} %, ties {ties})")

print("\n### full-budget cohort (seeds 84001-84010, 200-call snapshot of 3200-call runs), sealed c100 %, median / mean / worst")
print("| arm | single run, 800 calls | 4 runs x 200 calls, selected by training value | (reference) selected by sealed value (not a valid procedure) |\n|---|---|---|---|")
for arm, lab in (("eoh3200", "EoH"), ("h_improve", "steady-state T=0 + host"), ("h_improve_T0.001", "steady-state T=0.001 + host"), ("h_improve_T0.003", "steady-state T=0.003 + host")):
    seeds = list(range(84001, 84011)); key = {"eoh3200": "EoH3200", "h_improve": "T0", "h_improve_T0.001": "T0.001", "h_improve_T0.003": "T0.003"}[arm]
    tr = {s: {b: G7[f"{key}_{s}"]["train"][b] for b in ("200", "800")} for s in seeds}; se = {s: {b: val(SC[(arm, s)], b) for b in ("200", "800")} for s in seeds}
    single = [se[s]["800"] for s in seeds]; sel, orc = restart(tr, se, seeds, "200", 4); print(row(lab + ", 200x4", single, sel, orc))
eoh800 = [val(SC[("eoh3200", s)], "800") for s in range(84001, 84011)]
for arm, key, lab in (("h_improve", "T0", "T0"), ("h_improve_T0.003", "T0.003", "T0.003")):
    seeds = list(range(84001, 84011)); tr = {s: {"200": G7[f"{key}_{s}"]["train"]["200"]} for s in seeds}; se = {s: {"200": val(SC[(arm, s)], "200")} for s in seeds}
    sel, _ = restart(tr, se, seeds, "200", 4); wins = sum(x < y for x in sel for y in eoh800)
    print(f"  {lab} 200×4 (training-selected) < EoH 800: {wins}/{len(sel)*len(eoh800)} pairs ({100*wins/(len(sel)*len(eoh800)):.0f} %)")
