# Regenerate the tables (LaTeX) and figures (pgfplots) of arXiv v2 from the released per-run summaries
# (derived/grid7_readout.json, derived/readout_20261007.json) and the sealed scoring results (readouts/*.json).
# Usage: python paper_tables.py   -> writes analysis/out/tables/*.tex and analysis/out/figures/*.tex and prints the key numbers.
import sys
from pathlib import Path
D = Path(__file__).resolve().parents[1]  # experiments/hox_20261008
sys.path.insert(0, str(D / "scorers"))
import hoxN_eval_sealed as ev  # wilcoxon_exact, select
import json, statistics, itertools, importlib.util, collections
from pathlib import Path
P = D; OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
W = ev.wilcoxon_exact; med = statistics.median; mean = statistics.mean
G = json.load(open(D / "derived/grid7_readout.json", encoding="utf-8")); R9 = json.load(open(D / "derived/readout_20261007.json", encoding="utf-8"))["hox9"]
TS = ["T0", "T0.0003", "T0.001", "T0.003", "T0.01", "T0.03", "T0.1"]; TLAB = {"T0": "0", "T0.0003": "0.0003", "T0.001": "0.001", "T0.003": "0.003", "T0.01": "0.01", "T0.03": "0.03", "T0.1": "0.1"}
CA = {"EoH3200": "eoh3200", "T0": "h_improve", "T0.0003": "h_improve_t0003", "T0.001": "h_improve_T0.001", "T0.003": "h_improve_T0.003", "T0.01": "h_improve_T0.01", "T0.03": "h_improve_T0.03", "T0.1": "h_improve_T0.1"}
GA = {"T0": "g_T0", "T0.0003": "b_T0.0003", "T0.001": "b_T0.001", "T0.003": "g_T0003", "T0.01": "b_T0.01", "T0.03": "g_T003", "T0.1": "b_T0.1"}


def sealed(paths):
    by = {}
    for p in paths:
        for r in json.load(open(p, encoding="utf-8"))["runs"]: by[(r["arm"], r["seed"])] = r["at"]
    return by


S = D / "readouts"
SC = {b: sealed([S / f"hox2c_sealed_{b}.json", S / f"hox2ef_sealed_{b}.json", S / f"hox10c_sealed_{b}.json", S / f"hox11c_sealed_{b}.json"]) for b in ("c100", "c500")}
SG = {b: sealed([S / f"hox5_sealed_budgets_{b}.json", S / f"hox8_sealed_budgets_{b}.json", S / f"hox10b_sealed_budgets_{b}.json", S / f"hox11b_sealed_budgets_{b}.json"]) for b in ("c100", "c500")}
S9 = sealed([S / "hox9_sealed_c100.json"])


def sv(at, b):
    x = at.get(b) or at.get("end"); return 100 * x["value"]


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(idx): run = max(run, (len(ps) - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def spearman(xs, ys):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(s): r[i] = k + 1
        return r
    a, b = rk(xs), rk(ys); n = len(a); return 1 - 6 * sum((p - q) ** 2 for p, q in zip(a, b)) / (n * (n * n - 1))


def stats(v): return f"{med(v):.3f} & {mean(v):.3f} & {statistics.stdev(v):.3f} & {min(v):.3f} & {max(v):.3f}"


def fmt_p(p): return f"{p:.3f}" if p >= 0.001 else "$<$.001"


# ---------- per-run series
def cont_train(a, b): return [G["continuous"][f"{a}_{s}"]["train"][b] for s in range(84001, 84011)]
def gen_train(a, b): return [G["generational"][f"{a}_{s}"]["train"][b] for s in range(85001, 85011)]
def cont_seal(a, bank, b): return [sv(SC[bank][(CA[a], s)], b) for s in range(84001, 84011)]
def gen_seal(a, bank, b): return [sv(SG[bank][(GA[a], s)], b) for s in range(85001, 85011)]


# ---------- Table: temperature response (both families): training end + sealed c100 end, med/mean/sd/best/worst, paired vs T=0
def temp_table(fam, train, seal, label, caption):
    rows = []; ps_t = []; ps_s = []; wl_t = []; wl_s = []
    for a in TS[1:]:
        dt = [x - y for x, y in zip(train(a, "end"), train("T0", "end"))]; ds = [x - y for x, y in zip(seal(a, "c100", "end"), seal("T0", "c100", "end"))]
        ps_t.append(W(dt)); ps_s.append(W(ds)); wl_t.append(f"{sum(x<0 for x in dt)}/{sum(x>0 for x in dt)}"); wl_s.append(f"{sum(x<0 for x in ds)}/{sum(x>0 for x in ds)}")
    ht, hs = holm(ps_t), holm(ps_s)
    for i, a in enumerate(TS):
        if i == 0: test = "--- & --- & --- & ---"
        else: test = f"{wl_t[i-1]} & {fmt_p(ps_t[i-1])} & {wl_s[i-1]} & {fmt_p(ps_s[i-1])}"
        rows.append(f"{TLAB[a]} & {stats(train(a, 'end'))} & {stats(seal(a, 'c100', 'end'))} & {test}\\\\")
    body = "\n".join(rows)
    return (f"\\begin{{table*}}[t]\\centering\\footnotesize\n\\caption{{{caption}}}\\label{{{label}}}\n"
            "\\begin{tabular}{@{}l rrrrr rrrrr rrrr@{}}\\toprule\n"
            " & \\multicolumn{5}{c}{Training excess at the endpoint (\\%)} & \\multicolumn{5}{c}{Sealed c100 excess at the endpoint (\\%)} & \\multicolumn{4}{c}{Paired vs.\\ $T=0$ (W/L, $p$)}\\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-15}\n"
            "$T$ & Median & Mean & SD & Best & Worst & Median & Mean & SD & Best & Worst & Train & $p$ & Sealed & $p$\\\\\\midrule\n" + body + "\n\\bottomrule\\end{tabular}\\end{table*}\n")


(OUT / "tables").mkdir(exist_ok=True, parents=True); (OUT / "figures").mkdir(exist_ok=True, parents=True)


def temp_rows(train, seal):
    rows = []; ps_t = []; ps_s = []; wl_t = []; wl_s = []
    for a in TS[1:]:
        dt = [x - y for x, y in zip(train(a, "end"), train("T0", "end"))]; ds = [x - y for x, y in zip(seal(a, "c100", "end"), seal("T0", "c100", "end"))]
        ps_t.append(W(dt)); ps_s.append(W(ds)); wl_t.append(f"{sum(x<0 for x in dt)}/{sum(x>0 for x in dt)}"); wl_s.append(f"{sum(x<0 for x in ds)}/{sum(x>0 for x in ds)}")
    for i, a in enumerate(TS):
        test = "--- & --- & --- & ---" if i == 0 else f"{wl_t[i-1]} & {fmt_p(ps_t[i-1])} & {wl_s[i-1]} & {fmt_p(ps_s[i-1])}"
        rows.append(f"{TLAB[a]} & {stats(train(a, 'end'))} & {stats(seal(a, 'c100', 'end'))} & {test}\\\\")
    return "\n".join(rows)


(OUT / "tables/temperature.tex").write_text(
    "\\begin{table*}[t]\\centering\\footnotesize\n"
    "\\caption{Temperature response of the two families at the endpoint (10 runs per temperature; steady-state Fermi-type with the host on seeds 84001--84010, generational Bose-type with the host on seeds 85001--85010). "
    "Best and worst are single-run extremes and are reference values only. Paired columns: wins/losses of the given temperature against $T=0$ of the same family and the two-sided exact Wilcoxon $p$ (unadjusted; Holm-adjusted values are given in the text). "
    "The EoH reference (3,200 operator calls, seeds 84001--84010) has training 0.624 / 0.676 / 0.129 / 0.584 / 0.916 and sealed 0.711 / 0.811 / 0.245 / 0.617 / 1.273 in the same column order.}\\label{tab:temperature}\n"
    "\\begin{tabular}{@{}l rrrrr rrrrr rrrr@{}}\\toprule\n"
    " & \\multicolumn{5}{c}{Training excess (\\%)} & \\multicolumn{5}{c}{Sealed c100 excess (\\%)} & \\multicolumn{4}{c}{Paired vs.\\ $T=0$ (W/L, $p$)}\\\\\n"
    "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-15}\n"
    "$T$ & Median & Mean & SD & Best & Worst & Median & Mean & SD & Best & Worst & Train & $p$ & Sealed & $p$\\\\\\midrule\n"
    "\\multicolumn{15}{@{}l}{\\emph{Steady-state Fermi-type + host}}\\\\\n" + temp_rows(cont_train, cont_seal) + "\n\\midrule\n"
    "\\multicolumn{15}{@{}l}{\\emph{Generational Bose-type + host}}\\\\\n" + temp_rows(gen_train, gen_seal) + "\n\\bottomrule\\end{tabular}\\end{table*}\n", encoding="utf-8")

# EoH reference row (same seeds as the steady-state family) for the text
eoh_tr = cont_train("EoH3200", "end"); eoh_se = cont_seal("EoH3200", "c100", "end")
print("EoH3200 training end", stats(eoh_tr), "| sealed c100 end", stats(eoh_se))
print("EoH3200 training 800", stats(cont_train("EoH3200", "800")), "| sealed 800", stats(cont_seal("EoH3200", "c100", "800")))
print("T0+host training 800", stats(cont_train("T0", "800")), "| sealed 800", stats(cont_seal("T0", "c100", "800")))

# ---------- Table: diversity
rows = []
for a in TS:
    vc = [G["continuous"][f"{a}_{s}"] for s in range(84001, 84011)]; vg = [G["generational"][f"{a}_{s}"] for s in range(85001, 85011)]
    lc = collections.Counter()
    for x in vg: lc.update(x["lineage"])
    n = sum(lc.values())
    rows.append(f"{TLAB[a]} & {mean(x['pair'] for x in vc):.3f} & {mean(x['pats'] for x in vc):.1f} & {mean(x['spread'] for x in vc):.2f} & "
                f"{mean(x['distinct'] for x in vg):.1f} & {mean(x['pair'] for x in vg):.3f} & {mean(x['pats'] for x in vg):.1f} & {mean(x['spread'] for x in vg):.2f} & {mean(x['dsurv'] for x in vg):.2f} & {100*lc['no_best']/n:.0f}\\\\")
rho_c = spearman(list(range(7)), [mean(G["continuous"][f"{a}_{s}"]["spread"] for s in range(84001, 84011)) for a in TS])
rho_g = spearman(list(range(7)), [mean(G["generational"][f"{a}_{s}"]["spread"] for s in range(85001, 85011)) for a in TS])
(OUT / "tables/diversity.tex").write_text(
    "\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Final-population diversity and lineage use versus temperature (means over 10 runs). Code distance: mean pairwise token Jaccard distance; patterns: number of distinct feature patterns among the eight survivors; spread: objective range of the final population (percentage points); survivors: mean number of distinct survivor sources per generation during the run; non-best: share of new-best events whose parents excluded the current best.}\\label{tab:diversity}\n"
    "\\begin{tabular}{@{}l rrr rrrrrr@{}}\\toprule\n & \\multicolumn{3}{c}{Steady-state Fermi-type + host} & \\multicolumn{6}{c}{Generational Bose-type + host}\\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-10}\n"
    "$T$ & Code dist. & Patterns & Spread & Distinct/8 & Code dist. & Patterns & Spread & Survivors & Non-best (\\%)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n", encoding="utf-8")
print("spearman spread cont", f"{rho_c:+.3f}", "gen", f"{rho_g:+.3f}")

# ---------- Table: hox9 speed (sealed c100 at 200 / 400 / <=800), med/mean/best/worst + paired tests
arms9 = [("eoh800", "A: EoH, 800 operator calls"), ("h_improve", "B: Fermi-type $T=0$ + host"), ("h_improve_T0.003", "C: Fermi-type $T=0.003$ + host")]
rows = []
for a, lab in arms9:
    cells = []
    for b in ("200", "400", "800"):
        v = [sv(S9[(a, s)], b) for s in range(86001, 86011)]; cells.append(stats(v))
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
tests = []
for x, y, lab in (("h_improve", "eoh800", "B vs.\\ A"), ("h_improve_T0.003", "eoh800", "C vs.\\ A"), ("h_improve_T0.003", "h_improve", "C vs.\\ B")):
    cells = []
    for b in ("200", "400", "800"):
        d = [sv(S9[(x, s)], b) - sv(S9[(y, s)], b) for s in range(86001, 86011)]; cells.append(f"\\multicolumn{{5}}{{c}}{{{sum(v<0 for v in d)}/{sum(v>0 for v in d)}, $p={fmt_p(W(d))}$}}")
    tests.append(f"{lab} & " + " & ".join(cells) + "\\\\")
(OUT / "tables/speed_hox9.tex").write_text(
    "\\begin{table*}[t]\\centering\\footnotesize\n\\caption{Reach-speed confirmation (seeds 86001--86010, 10 runs per arm): sealed c100 excess (\\%) of the budget-point selection at 200, 400, and at most 800 total calls (operator plus host). The preregistered primary endpoint is B versus A at 400 total calls. Best and worst are single-run reference values. Paired rows: wins/losses of the first arm and two-sided exact Wilcoxon $p$.}\\label{tab:speed}\n"
    "\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{@{}l rrrrr rrrrr rrrrr@{}}\\toprule\n & \\multicolumn{5}{c}{200 total calls} & \\multicolumn{5}{c}{400 total calls (primary)} & \\multicolumn{5}{c}{$\\leq 800$ total calls (endpoint)}\\\\\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-16}\n"
    "Arm & Med. & Mean & SD & Best & Worst & Med. & Mean & SD & Best & Worst & Med. & Mean & SD & Best & Worst\\\\\\midrule\n" + "\n".join(rows) + "\n\\midrule\n" + "\n".join(tests) + "\n\\bottomrule\\end{tabular}\\end{table*}\n", encoding="utf-8")

# ---------- Figure: temperature response (pgfplots, symbolic x)
def series(fn, key, bank=None):
    out = []
    for a in TS:
        v = fn(a, bank, key) if bank else fn(a, key); q = statistics.quantiles(v, n=4); out.append((TLAB[a], med(v), q[0], q[2]))
    return out


def coords(ser): return " ".join(f"({t},{m:.4f}) +- (0,{max(m-lo,0):.4f}) +- (0,{max(hi-m,0):.4f})".replace("+- (0,", "+- (0,") for t, m, lo, hi in ser)


def pg_series(ser):  # asymmetric error bars: use error bars with explicit minus/plus via two tables is verbose; use IQR half-width approx -> explicit y error minus/plus
    return " ".join(f"({t},{m:.4f}) += (0,{hi-m:.4f}) -= (0,{m-lo:.4f})" for t, m, lo, hi in ser)


fig = r"""\begin{figure*}[t]\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=14mm},width=0.5\textwidth,height=5.6cm,
  symbolic x coords={0,0.0003,0.001,0.003,0.01,0.03,0.1},xtick=data,xlabel={Selection temperature $T$ (0 is a category; the rest are log-spaced)},
  ylabel={Excess (\%%), median with IQR},ymin=0.4,ymax=1.0,grid=major,legend style={font=\scriptsize,at={(0.03,0.97)},anchor=north west},
  error bars/y dir=both,error bars/y explicit,tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\small}]
\nextgroupplot[title={(a) Steady-state Fermi-type + host (seeds 84001--84010)}]
\addplot+[mark=*] coordinates {%s};\addlegendentry{training endpoint}
\addplot+[mark=square*] coordinates {%s};\addlegendentry{sealed c100 endpoint}
\nextgroupplot[title={(b) Generational Bose-type + host (seeds 85001--85010)}]
\addplot+[mark=*] coordinates {%s};\addlegendentry{training endpoint}
\addplot+[mark=square*] coordinates {%s};\addlegendentry{sealed c100 endpoint}
\end{groupplot}
\end{tikzpicture}
\caption{Temperature response of the two families. Each point is the median over 10 seed-paired runs; bars show the interquartile range. The steady-state family (a) is best at $T=0$ and degrades slowly with temperature; the generational family (b) collapses at $T=0$ and is best between $T=0.001$ and $0.003$. The two families use different seeds and are compared only within a panel.}\label{fig:temperature}
\end{figure*}
""" % (pg_series(series(cont_train, "end")), pg_series(series(cont_seal, "end", "c100")), pg_series(series(gen_train, "end")), pg_series(series(gen_seal, "end", "c100")))
(OUT / "figures/temperature_response.tex").write_text(fig, encoding="utf-8")

# ---------- Figure: diversity (objective spread, log y) and code distance
def dser(fam, key, seeds):
    return " ".join(f"({TLAB[a]},{mean(G[fam][f'{a}_{s}'][key] for s in seeds):.4f})" for a in TS)


fig2 = r"""\begin{figure}[t]\centering
\begin{tikzpicture}
\begin{axis}[width=\columnwidth,height=5.2cm,symbolic x coords={0,0.0003,0.001,0.003,0.01,0.03,0.1},xtick=data,ymode=log,
  xlabel={Selection temperature $T$},ylabel={Objective spread of the final population (pp)},grid=major,legend style={font=\scriptsize,at={(0.03,0.97)},anchor=north west},
  tick label style={font=\scriptsize},label style={font=\scriptsize},ymin=0.03,ymax=10]
\addplot+[mark=*] coordinates {%s};\addlegendentry{steady-state Fermi-type + host}
\addplot+[mark=square*] coordinates {%s};\addlegendentry{generational Bose-type + host}
\end{axis}
\end{tikzpicture}
\caption{Diversity of the retained population versus temperature (mean over 10 runs; log vertical axis). The spread is the range of training excess within the final population of eight; the generational $T=0$ value is zero (all copies) and is drawn at the axis floor.}\label{fig:diversity}
\end{figure}
""" % (dser("continuous", "spread", range(84001, 84011)).replace("(0,0.0000)", "(0,0.03)"), dser("generational", "spread", range(85001, 85011)).replace("(0,0.0000)", "(0,0.03)"))
(OUT / "figures/diversity.tex").write_text(fig2, encoding="utf-8")

# ---------- Figure: reach speed, sealed c100 median vs total calls (hox2c family, 3200 runs) + hox9 panel
def budget_series(getter, arm, budgets, xs):
    return " ".join(f"({x},{med([getter(arm, s, b) for s in SEEDS]):.4f})" for x, b in zip(xs, budgets))


SEEDS = range(84001, 84011)
g2c = lambda arm, s, b: sv(SC["c100"][(CA[arm], s)], b)
s2c = {a: budget_series(g2c, a, ("200", "400", "800", "1600", "end"), (200, 400, 800, 1600, 3200)) for a in ("EoH3200", "T0", "T0.0003", "T0.003")}
SEEDS = range(86001, 86011)
g9 = lambda arm, s, b: sv(S9[(arm, s)], b)
s9 = {a: budget_series(g9, a, ("200", "400", "800"), (200, 400, 800)) for a in ("eoh800", "h_improve", "h_improve_T0.003")}
fig3 = r"""\begin{figure*}[t]\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=14mm},width=0.5\textwidth,height=5.6cm,xmode=log,ymode=log,log basis x=2,
  xlabel={Total LLM calls (operator + host), log scale},ylabel={Sealed c100 excess (\%%), median},grid=major,legend style={font=\scriptsize,at={(0.97,0.97)},anchor=north east},
  tick label style={font=\scriptsize},label style={font=\scriptsize},title style={font=\small}]
\nextgroupplot[title={(a) Full-budget cohort (seeds 84001--84010)},xtick={200,400,800,1600,3200},xticklabels={200,400,800,1600,3200}]
\addplot+[mark=triangle*] coordinates {%s};\addlegendentry{EoH, 3200 operator calls}
\addplot+[mark=*] coordinates {%s};\addlegendentry{Fermi-type $T=0$ + host}
\addplot+[mark=square*] coordinates {%s};\addlegendentry{Fermi-type $T=0.0003$ + host}
\addplot+[mark=diamond*] coordinates {%s};\addlegendentry{Fermi-type $T=0.003$ + host}
\nextgroupplot[title={(b) Preregistered confirmation (seeds 86001--86010)},xtick={200,400,800},xticklabels={200,400,$\leq$800}]
\addplot+[mark=triangle*] coordinates {%s};\addlegendentry{A: EoH, 800 operator calls}
\addplot+[mark=*] coordinates {%s};\addlegendentry{B: Fermi-type $T=0$ + host}
\addplot+[mark=diamond*] coordinates {%s};\addlegendentry{C: Fermi-type $T=0.003$ + host}
\end{groupplot}
\end{tikzpicture}
\caption{Reach speed on the sealed confirmation bank. Each point is the median over 10 runs of the budget-point selection (best training individual among registrations within the call budget). Host calls are counted in the budget. In (b) the host arms stop at or below 800 total calls, so their last point is the endpoint.}\label{fig:speed}
\end{figure*}
""" % (s2c["EoH3200"], s2c["T0"], s2c["T0.0003"], s2c["T0.003"], s9["eoh800"], s9["h_improve"], s9["h_improve_T0.003"])
(OUT / "figures/speed.tex").write_text(fig3, encoding="utf-8")

# ---------- numbers for the text
for a in ("eoh800", "h_improve", "h_improve_T0.003"):
    for b in ("200", "400", "800"):
        v = [sv(S9[(a, s)], b) for s in range(86001, 86011)]; print("hox9", a, b, f"med {med(v):.3f} mean {mean(v):.3f} min {min(v):.3f} max {max(v):.3f}")
for a in TS:
    print("gen", a, "train end", stats(gen_train(a, "end")), "| sealed", stats(gen_seal(a, "c100", "end")), "| c500 med", f"{med(gen_seal(a, 'c500', 'end')):.3f}")
for a in TS:
    print("cont", a, "train end", stats(cont_train(a, "end")), "| sealed", stats(cont_seal(a, "c100", "end")), "| c500 med", f"{med(cont_seal(a, 'c500', 'end')):.3f}")
print("written", sorted(p.name for p in (OUT / "tables").iterdir()), sorted(p.name for p in (OUT / "figures").iterdir()))


