"""Sealed held-out readout of the generational cohorts at call budgets (CPU only). Budget b -> generations g = b // 18
(nominal 9 operator + 9 host calls per generation); the selected individual is the lowest-training-energy member of the
lineage with gen <= g (gen 0 = bank at 0 calls); ties -> earliest id. Same scorer and worst-case rule as hoxN_eval_sealed.py.
usage (from the repository root): python experiments/hox_20261008/scorers/hox5_eval_sealed_budgets.py --bank B --seal S --out O COHORT_DIR [COHORT_DIR ...]"""
import argparse, hashlib, json, re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hoxN_eval_sealed as H  # noqa: E402
LABEL = re.compile(r"_(g_T0|g_T0003|g_T003|b_T[0-9.]+)_cs([0-9]+)$")
BUDGETS = (200, 400, 800, 1600, "end")
ap = argparse.ArgumentParser(); ap.add_argument("cohorts", nargs="+"); ap.add_argument("--bank", required=True); ap.add_argument("--seal", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
raw = Path(a.bank).read_bytes()
sealed = dict(l.split(" sha256 ") for l in Path(a.seal).read_text(encoding="utf-8").splitlines() if l)
if sealed.get(Path(a.bank).name) != hashlib.sha256(raw).hexdigest():
    sys.exit("REFUSED: bank does not match its seal")
bank = json.loads(raw)
from get_instance import GetData  # noqa: E402
from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: E402
gd = GetData()
worst = statistics.mean((int(v["num_items"]) - (lb := float(gd.l1_bound(tuple(int(x) for x in v["items"]), int(v["capacity"]))))) / lb for v in bank["instances"].values())
adapter = BpOnlineAdapter(train_k=5, items=5000, eval_timeout=30.0, deterministic_only=True)
cache, runs = {}, []
for c in a.cohorts:
    for d in sorted(Path(c).iterdir()):
        m = LABEL.search(d.name)
        if not (d.is_dir() and m):
            continue
        res = json.load(open(next((d / "r1/tgade").glob("result_*.json")), encoding="utf-8"))
        lin = [x for x in res["lineage"] if x.get("energy") is not None]
        G = max(x["gen"] for x in lin)
        rec = {"label": d.name, "arm": m.group(1), "seed": int(m.group(2)), "generations": G, "at": {}}
        for b in BUDGETS:
            g = G if b == "end" else b // 18
            if g > G:
                rec["at"][str(b)] = None; continue
            best = min((x for x in lin if x["gen"] <= g), key=lambda x: (x["energy"], x["id"]))
            code = best["genes"]["code"]; sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if sha not in cache:
                per = adapter.endpoint_raw({"code": code}, bank)
                cache[sha] = (worst, True) if per is None else (statistics.mean(per.values()), False)
            val, failed = cache[sha]
            rec["at"][str(b)] = {"gen": g, "train_pct": 200 * best["energy"], "code_sha256": sha, "value": val, "failed": failed}
        runs.append(rec)
        print(rec["label"], {b: (None if v is None else round(100 * v["value"], 3)) for b, v in rec["at"].items()}, flush=True)
by = {(r["arm"], r["seed"]): r for r in runs}
arms = sorted({r["arm"] for r in runs}); tests = []
for x in arms:
    for y in arms:
        if x == y: continue
        for b in BUDGETS:
            pr = [(by[(x, s)]["at"][str(b)]["value"], by[(y, s)]["at"][str(b)]["value"]) for (arm, s) in by if arm == x and (y, s) in by
                  and by[(x, s)]["at"][str(b)] is not None and by[(y, s)]["at"][str(b)] is not None]
            if pr:
                diff = [p - q for p, q in pr]
                tests.append({"pair": f"{x} vs {y}", "budget": str(b), "n": len(pr), "wins": sum(v < 0 for v in diff), "losses": sum(v > 0 for v in diff),
                              "p": H.wilcoxon_exact(diff), "median_x_pct": round(100 * statistics.median(p for p, _ in pr), 4), "median_y_pct": round(100 * statistics.median(q for _, q in pr), 4)})
for t in tests: print(t)
Path(a.out).write_text(json.dumps({"bank": Path(a.bank).name, "bank_sha256": sealed[Path(a.bank).name], "worst_case": worst, "budget_rule": "gen = budget // 18",
                                   "runs": runs, "tests": tests}, indent=1), encoding="utf-8")
