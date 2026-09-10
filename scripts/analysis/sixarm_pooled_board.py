import os
"""generational scoreboard vs the steady-state configurations on the pooled seeds 20501-20510 + 20601-20610 (n=20).
Endpoints: final.json (generational: final-population best; steady-state: history best, unrounded) and g1_history_best.json."""
import glob, json, os, re, statistics as st
ROOT = os.environ.get("TGADE_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "runs", "v101"))
G1 = {"G1-fermion0": "fermion_T0", "G1-boson.003": "boson_T0.003", "G1-boson.03": "boson_T0.03"}
R = {"EoH": "main_eoh_800_evt30_s2e2", "L": "x1_level_T0.00001_800_bh_sigE_s2e2", "N": "x1_none_T0.003_800_bh_sigE_s2e2"}


def q(v, p):
    v = sorted(v); k = (len(v) - 1) * p; f = int(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)


ref = {}
for arm, pre in R.items():
    for d in glob.glob(os.path.join(ROOT, f"eoh_{pre}_cs20[56]*_2026*")):
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp, encoding="utf-8")); cs = int(d.split("_cs")[1][:5])
        fj = json.load(open(os.path.join(d, "final.json"), encoding="utf-8"))
        ref[(arm, cs)] = {"train": s.get("train_raw_excess_best_unclipped", s["history_best_objective"]), "c100": float(fj["c100"]), "c500": float(fj["c500"]),
                          "usd": s["usd"], "h": s["elapsed_s"] / 3600}
rows = {}
for arm, tag in G1.items():
    for f in sorted(glob.glob(os.path.join(ROOT, f"g1_x2full_bh_sigE_{tag}_evt30_cs20[56]*_2026*"))):
        fj = json.load(open(os.path.join(f, "final.json"), encoding="utf-8"))
        m = type("M", (), {"group": staticmethod(lambda i, fj=fj: [None, fj["train_raw_excess_best"], fj["c100"], fj["c500"], fj["calls"], fj["elapsed_s"], fj["usd"]][i])})()
        cs = int(re.search(r"_cs(\d+)", f).group(1))
        hb = glob.glob(os.path.join(ROOT, f"g1_x2full_bh_sigE_{tag}_evt30_cs{cs}_2026*", "g1_history_best.json"))
        hbj = json.load(open(hb[0], encoding="utf-8")) if hb else {}
        rows.setdefault(arm, []).append({"cs": cs, "train": float(m.group(1)), "c100": float(m.group(2)), "c500": float(m.group(3)),
                                         "calls": int(m.group(4)), "h": float(m.group(5)) / 3600, "usd": float(m.group(6)),
                                         "hb_equal": (hbj.get("history_best_energy") == hbj.get("final_best_energy")) if hbj else None,
                                         "hb_c500": hbj.get("history_best_c500")})
print(f"{'arm':13s} {'n':>2s} {'train med':>9s} {'IQR':>6s} {'sd':>6s} {'c100':>6s} {'c500':>6s} {'stall':>6s} {'collap':>6s} {'vsEoH':>6s} {'vsN':>6s} {'c500vsEoH':>9s} {'calls':>5s} {'usd':>6s} {'h':>5s} {'hb=fin':>6s}")
for arm, v in rows.items():
    tr = [x["train"] for x in v]; c1 = [x["c100"] for x in v]; c5 = [x["c500"] for x in v]
    we = le = wn = ln = w5 = l5 = 0
    for x in v:
        e = ref.get(("EoH", x["cs"])); n = ref.get(("N", x["cs"]))
        if e: we += x["train"] < e["train"]; le += x["train"] > e["train"]; w5 += x["c500"] < e["c500"]; l5 += x["c500"] > e["c500"]
        if n: wn += x["train"] < n["train"]; ln += x["train"] > n["train"]
    print(f"{arm:13s} {len(v):2d} {100*st.median(tr):9.3f} {100*(q(tr,.75)-q(tr,.25)):6.3f} {100*st.pstdev(tr):6.3f} {100*st.median(c1):6.3f} {100*st.median(c5):6.3f} "
          f"{sum(x>0.02 for x in tr):>3d}/{len(v):<2d} {sum(x>0.1 for x in c5):>3d}/{len(v):<2d} {we}-{le:<3d} {wn}-{ln:<3d} {w5}-{l5:<6d} {st.median(x['calls'] for x in v):5.0f} {st.median(x['usd'] for x in v):6.3f} {st.median(x['h'] for x in v):5.2f} {sum(bool(x['hb_equal']) for x in v)}/{len(v)}")
for arm, pre in R.items():
    v = [ref[k] for k in ref if k[0] == arm]; tr = [x["train"] for x in v]; c5 = [x["c500"] for x in v]
    print(f"{'R-'+arm:13s} {len(v):2d} {100*st.median(tr):9.3f} {100*(q(tr,.75)-q(tr,.25)):6.3f} {100*st.pstdev(tr):6.3f} {100*st.median(x['c100'] for x in v):6.3f} {100*st.median(c5):6.3f} "
          f"{sum(x>0.02 for x in tr):>3d}/{len(v):<2d} {sum(x>0.1 for x in c5):>3d}/{len(v):<2d} {'':6s} {'':6s} {'':9s} {800:5d} {st.median(x['usd'] for x in v):6.3f} {st.median(x['h'] for x in v):5.2f}")
print("\nper-seed (train %): seed | R-EoH | R-N | G1 fermion | G1 boson.003 | G1 boson.03")
for cs in list(range(20501, 20511)) + list(range(20601, 20611)):
    g = {arm: next((x for x in rows.get(arm, []) if x["cs"] == cs), None) for arm in G1}
    print(cs, "|", " | ".join(f"{100*ref[(a,cs)]['train']:.3f}" if (a, cs) in ref else "-" for a in ("EoH", "N")), "|",
          " | ".join(f"{100*g[a]['train']:.3f}/{100*g[a]['c500']:.2f}" if g[a] else "-" for a in G1))
