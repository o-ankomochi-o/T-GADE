"""Figure 2 data (best-so-far training excess vs logical calls; median and IQR over runs) for 6 main arms (pooled seeds
20501-20510 + 20601-20610) and B1/B2 (20601-20610); Table VI data (operator-wise offspring outcome shares) for the
three generational arms. $0: reads run records only."""
import glob, json, os, statistics as st, collections
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("TGADE_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "runs", "v101"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "paper_v1")
GRID = list(range(9, 801, 9)) + [800]
STEADY = {"EoH": "eoh_main_eoh_800_evt30_s2e2_cs20[56]*_2026*", "L (T=1e-5, level)": "eoh_x1_level_T0.00001_800_bh_sigE_s2e2_cs20[56]*_2026*",
          "N (T=0.003, unrestricted)": "eoh_x1_none_T0.003_800_bh_sigE_s2e2_cs20[56]*_2026*",
          "B1 parent-free": "eoh_b1_i1only_800_evt30_s2e2_cs206*_2026*", "B2 quality-blind": "eoh_b2_newest_uniform_800_evt30_s2e2_cs206*_2026*"}
GEN = {"Gen. fermion T=0": "g1_x2full_bh_sigE_fermion_T0_evt30_cs20[56]*_2026*", "Gen. boson T=0.003": "g1_x2full_bh_sigE_boson_T0.003_evt30_cs20[56]*_2026*",
       "Gen. boson T=0.03": "g1_x2full_bh_sigE_boson_T0.03_evt30_cs20[56]*_2026*"}

def curve_steady(d):
    pts = []
    for l in open(os.path.join(d, "registration.jsonl"), encoding="utf-8"):
        r = json.loads(l); nc = r.get("newcomer") or {}
        if nc.get("objective") is not None and r.get("call_seq") is not None:
            pts.append((r["call_seq"], 100 * nc["objective"]))
    pts.sort(); out = []; best = float("inf"); i = 0
    for n in GRID:
        while i < len(pts) and pts[i][0] <= n:
            best = min(best, pts[i][1]); i += 1
        out.append(best if best < float("inf") else None)
    return out

def curve_gen(d):
    o = json.load(open(os.path.join(d, "r1", "tgade", "result_bp_online_seed101.json"), encoding="utf-8"))
    bygen = collections.defaultdict(lambda: float("inf"))
    for x in o["lineage"]:
        e = x.get("energy")
        if isinstance(e, (int, float)):
            bygen[int(x.get("gen", 0))] = min(bygen[int(x.get("gen", 0))], 200 * e)
    out = []; best = float("inf")
    for n in GRID:
        g = n // 9
        for gg in range(0, g + 1):
            best = min(best, bygen.get(gg, float("inf")))
        out.append(best if best < float("inf") else None)
    return out

fig2 = {"calls": GRID, "arms": {}}
for arm, pat in list(STEADY.items()) + list(GEN.items()):
    dirs = sorted(glob.glob(os.path.join(ROOT, pat)))
    curves = [curve_gen(d) if arm.startswith("Gen.") else curve_steady(d) for d in dirs]
    med, q25, q75 = [], [], []
    for j in range(len(GRID)):
        v = [c[j] for c in curves if c[j] is not None]
        med.append(float(np.median(v)) if v else None); q25.append(float(np.percentile(v, 25)) if v else None); q75.append(float(np.percentile(v, 75)) if v else None)
    fig2["arms"][arm] = {"n": len(curves), "median": med, "q25": q25, "q75": q75}
    print(f"{arm:28s} n={len(curves):2d} best-so-far median % at calls 90/180/360/540/720/800: " + " ".join(f"{med[GRID.index(c)]:.3f}" for c in (90, 180, 360, 540, 720, 800)))
json.dump(fig2, open(os.path.join(OUT, "fig2_best_so_far_20260909.json"), "w", encoding="utf-8"))
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, o in fig2["arms"].items():
    ax.plot(GRID, o["median"], label=arm, linestyle="--" if arm.startswith("Gen.") else ("-" if not arm.startswith("B") else ":"))
ax.set_yscale("log"); ax.set_xlabel("logical generation calls"); ax.set_ylabel("best-so-far training excess (%)  [median over runs]"); ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_best_so_far_preview_20260909.png"), dpi=120)

# Table VI
OPS = ["eoh_e1", "eoh_e2", "eoh_m1", "eoh_m2"]; t6 = {}
print("\nTable VI: arm | op | pairs | better% | equal% | worse% | child median % | best-parent median % | pooled rho (tie-aware)")
for arm, pat in GEN.items():
    rows = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        o = json.load(open(os.path.join(d, "r1", "tgade", "result_bp_online_seed101.json"), encoding="utf-8"))
        byid = {x["id"]: x for x in o["lineage"]}
        for x in o["lineage"]:
            e = x.get("energy"); op = x.get("op")
            if not isinstance(e, (int, float)) or op not in OPS or not x.get("parent_ids"):
                continue
            pe = [byid[p]["energy"] for p in x["parent_ids"] if p in byid and isinstance(byid[p].get("energy"), (int, float))]
            if len(pe) != len(x["parent_ids"]) or not pe:
                continue
            rows[op].append((200 * min(pe), 200 * e))
    for op in OPS:
        v = np.array(rows[op]); bp, c = v[:, 0], v[:, 1]
        better = float(np.mean(c < bp - 1e-9)); equal = float(np.mean(np.abs(c - bp) <= 1e-9)); worse = 1 - better - equal
        rho = float(spearmanr(bp, c).correlation)
        t6[f"{arm}|{op}"] = {"pairs": int(len(v)), "better": better, "equal": equal, "worse": worse, "child_median": float(np.median(c)),
                            "best_parent_median": float(np.median(bp)), "rho_pooled_tie_aware": rho, "runs": 20}
        print(f"{arm:20s} {op[4:]:3s} {len(v):5d} {100*better:6.1f} {100*equal:6.1f} {100*worse:6.1f} {np.median(c):8.3f} {np.median(bp):8.3f} {rho:+.2f}")
json.dump(t6, open(os.path.join(OUT, "table6_operator_outcomes_20260909.json"), "w", encoding="utf-8"), indent=1)

# --- per-run exports for the public release (every figure/table reproducible from released data) ---
per_run_curves = {}
for arm, pat in list(STEADY.items()) + list(GEN.items()):
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        per_run_curves[os.path.basename(d)] = {"arm": arm, "calls": GRID, "best_so_far_pct": (curve_gen(d) if arm.startswith("Gen.") else curve_steady(d))}
json.dump(per_run_curves, open(os.path.join(OUT, "per_run_best_so_far_20260909.json"), "w", encoding="utf-8"))
per_run_ops = {}
for arm, pat in GEN.items():
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        o = json.load(open(os.path.join(d, "r1", "tgade", "result_bp_online_seed101.json"), encoding="utf-8"))
        byid = {x["id"]: x for x in o["lineage"]}; rec = {}
        for op in OPS:
            b = e = w = 0; ch = []; par = []
            for x in o["lineage"]:
                en = x.get("energy")
                if not isinstance(en, (int, float)) or x.get("op") != op or not x.get("parent_ids"):
                    continue
                pe = [byid[p]["energy"] for p in x["parent_ids"] if p in byid and isinstance(byid[p].get("energy"), (int, float))]
                if len(pe) != len(x["parent_ids"]) or not pe:
                    continue
                bp, c = 200 * min(pe), 200 * en; ch.append(c); par.append(bp)
                if c < bp - 1e-9: b += 1
                elif abs(c - bp) <= 1e-9: e += 1
                else: w += 1
            rec[op] = {"pairs": b + e + w, "better": b, "equal": e, "worse": w, "child_median_pct": float(np.median(ch)) if ch else None, "best_parent_median_pct": float(np.median(par)) if par else None}
        per_run_ops[os.path.basename(d)] = {"arm": arm, "operators": rec}
json.dump(per_run_ops, open(os.path.join(OUT, "per_run_operator_outcomes_20260909.json"), "w", encoding="utf-8"), indent=0)
print("per-run exports:", len(per_run_curves), "curves,", len(per_run_ops), "operator records")
