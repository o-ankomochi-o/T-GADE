# Median / mean / SD / best / worst over the 10 seeds per arm: training excess (budget 800 and endpoint) and sealed c100 / c500.
# Usage: python stats.py  -> prints Markdown tables and writes analysis/out/stats.md
import sys
from pathlib import Path
D = Path(__file__).resolve().parents[1]  # experiments/hox_20261008
sys.path.insert(0, str(D / "scorers"))
import hoxN_eval_sealed as ev  # wilcoxon_exact, select
import json, statistics
from pathlib import Path
P = D; G = json.load(open(D / "derived/grid7_readout.json", encoding="utf-8"))
TS = ["T0", "T0.0003", "T0.001", "T0.003", "T0.01", "T0.03", "T0.1"]


def load_runs(paths):
    by = {}
    for p in paths:
        for r in json.load(open(p, encoding="utf-8"))["runs"]: by[(r["arm"], r["seed"])] = r["at"]
    return by


S = D / "readouts"
CA = [("EoH3200", "eoh3200"), ("T0", "h_improve"), ("T0.0003", "h_improve_t0003"), ("T0.001", "h_improve_T0.001"), ("T0.003", "h_improve_T0.003"), ("T0.01", "h_improve_T0.01"), ("T0.03", "h_improve_T0.03"), ("T0.1", "h_improve_T0.1")]
GA = [("T0", "g_T0"), ("T0.0003", "b_T0.0003"), ("T0.001", "b_T0.001"), ("T0.003", "g_T0003"), ("T0.01", "b_T0.01"), ("T0.03", "g_T003"), ("T0.1", "b_T0.1")]
sealed = {}
for bank in ("c100", "c500"):
    sealed[("c", bank)] = load_runs([S / f"hox2c_sealed_{bank}.json", S / f"hox2ef_sealed_{bank}.json", S / f"hox10c_sealed_{bank}.json", S / f"hox11c_sealed_{bank}.json"])
    sealed[("g", bank)] = load_runs([S / f"hox5_sealed_budgets_{bank}.json", S / f"hox8_sealed_budgets_{bank}.json", S / f"hox10b_sealed_budgets_{bank}.json", S / f"hox11b_sealed_budgets_{bank}.json"])


def st(v):
    v = [x for x in v if x is not None]
    return f"{statistics.median(v):.3f} | {statistics.mean(v):.3f} | {statistics.stdev(v) if len(v) > 1 else 0:.3f} | {min(v):.3f} | {max(v):.3f} | {len(v)}"


HDR = "| arm | median | mean | SD | best (min) | worst (max) | n |\n|---|---|---|---|---|---|---|"
out = []
for fam, key, arms, seeds, famname in (("c", "continuous", CA, range(84001, 84011), "EoH 3200 (no host) and steady-state Fermi-type + host"), ("g", "generational", GA, range(85001, 85011), "generational Bose-type + host")):
    for metric, getter in (("training, 800 total calls", lambda a, s: G[key][f"{a}_{s}"]["train"]["800"] if f"{a}_{s}" in G[key] else None),
                           ("training, endpoint", lambda a, s: G[key][f"{a}_{s}"]["train"]["end"] if f"{a}_{s}" in G[key] else None),
                           ("sealed c100, 800 total calls", lambda a, s: 100 * sealed[(fam, "c100")][(ARM[a], s)]["800"]["value"] if (ARM[a], s) in sealed[(fam, "c100")] and sealed[(fam, "c100")][(ARM[a], s)]["800"] else None),
                           ("sealed c100, endpoint", lambda a, s: 100 * sealed[(fam, "c100")][(ARM[a], s)]["end"]["value"] if (ARM[a], s) in sealed[(fam, "c100")] else None),
                           ("sealed c500, endpoint", lambda a, s: 100 * sealed[(fam, "c500")][(ARM[a], s)]["end"]["value"] if (ARM[a], s) in sealed[(fam, "c500")] else None)):
        ARM = dict(arms)
        out.append(f"\n### {famname}: {metric} (excess %, lower is better, seeds {seeds.start}-{seeds.stop - 1})\n\n{HDR}")
        for lab, a in arms:
            out.append(f"| {lab} | {st([getter(lab, s) for s in seeds])} |")
txt = "\n".join(out); print(txt); open(D / "analysis/out/stats.md", "w", encoding="utf-8").write(txt + "\n")
