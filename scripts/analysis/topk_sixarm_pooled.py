import os
"""Top-k selection curves (P1 / P2 / P3) from per-run pop_endpoint.json for the steady-state configurations and G1
(generational), same seeds 20601-20610. P2 = select among top-k by train with one C500 validation instance
(first key), test on the other four; P3 = oracle min of the 5-instance mean. Figure: six arms on one panel (P2)."""
import glob, json, os, statistics as st
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "paper_v1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("TGADE_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "runs", "v101"))
ARMS = {
    "EoH (steady)": "eoh_main_eoh_800_evt30_s2e2_cs20[56]*_2026*",
    "L 1e-5 (steady, EoH replicate)": "eoh_x1_level_T0.00001_800_bh_sigE_s2e2_cs20[56]*_2026*",
    "N 0.003 (steady)": "eoh_x1_none_T0.003_800_bh_sigE_s2e2_cs20[56]*_2026*",
    "G1 fermion T=0 (gen.)": "g1_x2full_bh_sigE_fermion_T0_evt30_cs20[56]*_2026*",
    "G1 boson 0.003 (gen.)": "g1_x2full_bh_sigE_boson_T0.003_evt30_cs20[56]*_2026*",
    "G1 boson 0.03 (gen.)": "g1_x2full_bh_sigE_boson_T0.03_evt30_cs20[56]*_2026*",
}
out = {}
print(f"{'arm':32s} {'n':>2s} {'P1':>6s} {'P2k1':>6s} {'P2k2':>6s} {'P2k4':>6s} {'P2k8':>6s} {'P3k8':>6s} {'pop>10%':>7s} {'pop<1%':>6s} {'pop med':>7s}")
for arm, pat in ARMS.items():
    P1, P2, P3, pop = [], {k: [] for k in range(1, 9)}, {k: [] for k in range(1, 9)}, []
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        pe = os.path.join(d, "pop_endpoint.json")
        if not os.path.exists(pe):
            continue
        rows = json.load(open(pe, encoding="utf-8"))
        rr = []
        for r in rows:
            inst = r.get("c500_inst") or {}
            ks = sorted(inst)
            val = inst[ks[0]] if ks else r["c500"]
            test = st.mean(inst[k] for k in ks[1:]) if len(ks) > 1 else r["c500"]
            rr.append({"train": r["train"], "c500": r["c500"], "val": val, "test": test})
        if not rr:
            continue
        rr.sort(key=lambda r: r["train"])
        P1.append(rr[0]["test"]); pop += [r["c500"] for r in rr]
        for k in range(1, 9):
            top = rr[:k]
            P2[k].append(min(top, key=lambda r: r["val"])["test"]); P3[k].append(min(r["c500"] for r in top))
    if not P1:
        continue
    out[arm] = {"n": len(P1), "P1": P1, "P2": P2, "P3": P3, "pop": pop}
    print(f"{arm:32s} {len(P1):2d} {100*st.median(P1):6.3f} {100*st.median(P2[1]):6.3f} {100*st.median(P2[2]):6.3f} {100*st.median(P2[4]):6.3f} {100*st.median(P2[8]):6.3f} {100*st.median(P3[8]):6.3f} "
          f"{sum(v>0.1 for v in pop)/len(pop):7.2f} {sum(v<0.01 for v in pop)/len(pop):6.2f} {100*st.median(pop):7.3f}")
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm, o in out.items():
    ks = list(range(1, 9)); ax.plot(ks, [100 * st.median(o["P2"][k]) for k in ks], marker="o", label=arm, linestyle="--" if "gen." in arm else "-")
ax.set_xlabel("k (top-k by C100 train objective)"); ax.set_ylabel("median test C500 excess (%) over seeds"); ax.set_yscale("log"); ax.grid(alpha=.3)
ax.set_title("P2: validation-selected among top-k, pooled n=20 (seeds 20501-20510 + 20601-20610)", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); png = os.path.join(OUT_DIR, "tgade_sixarm_pooled_topk_c500_20260909.png"); fig.savefig(png, dpi=130)
json.dump({a: {"n": o["n"], "P1": o["P1"], "P2": o["P2"], "P3": o["P3"]} for a, o in out.items()},
          open(os.path.join(OUT_DIR, "tgade_sixarm_pooled_topk_c500_20260909.json"), "w", encoding="utf-8"), indent=1)
print("saved", png)
