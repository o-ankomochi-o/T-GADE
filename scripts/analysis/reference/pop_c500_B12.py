import os
"""Population-level generalisation:
evaluate every member of the FINAL population of each s2e2 cohort run on the sealed C100 and C500
confirmation banks (same endpoint evaluator as the selected-best endpoint). Output per run
pop_endpoint.json and an arm-level table (median / min / fraction >10% of population c500)."""
import glob, hashlib, importlib.util, json, os, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor

REPO = os.environ.get("TGADE_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/src")
spec = importlib.util.spec_from_file_location("run_v101_eoh_pop", REPO + "/scripts/run_v101_eoh.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: E402

ROOT = os.environ.get("TGADE_RUNS", os.path.join(REPO, "results", "runs", "v101"))
ARMS = {"B1": "b1_i1only_800_evt30_s2e2", "B2": "b2_newest_uniform_800_evt30_s2e2"}
adapter = BpOnlineAdapter(train_k=5, items=5000, objective="train_c100", diversity_carrier="behaviour",
                          probe_len=64, eval_timeout=90, signature_source="probe", deterministic_only=False)
banks = {"c100": m._load_bank("confirmation_bank.json"), "c500": m._load_bank("confirmation_bank_c500.json")}
penalties = {k: m._bank_penalty(b) for k, b in banks.items()}
CACHE_PATH = os.path.join(ROOT, "pop_endpoint_cache_B12_20260909.json")  # disk cache: reruns are free
cache = json.load(open(CACHE_PATH, encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
import threading
_clock = threading.Lock()


def ep(code, thought):
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]
    r = m._endpoint(adapter, {"code": code, "thought": thought or ""}, banks, penalties)
    rec = {"c100": r["c100"]["raw"], "c500": r["c500"]["raw"], "penalty": r["c500"]["penalty"] or r["c100"]["penalty"],
           "c100_inst": r["c100"]["per_instance"], "c500_inst": r["c500"]["per_instance"]}
    with _clock:
        cache[key] = rec
        json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"))
    return rec


runs = []
for arm, pre in ARMS.items():
    for d in sorted(glob.glob(os.path.join(ROOT, f"eoh_{pre}_cs206*_2026*"))):
        if "ABANDONED" in d or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        fs = sorted(glob.glob(os.path.join(d, "results", "pops", "population_generation_*.json")),
                    key=lambda p: int(p.rsplit("_", 1)[1][:-5]))
        pop = json.load(open(fs[-1], encoding="utf-8"))
        runs.append((arm, d, pop))
print("runs", len(runs), "individuals", sum(len(p) for _, _, p in runs), "unique codes",
      len({hashlib.sha256(x["code"].encode("utf-8")).hexdigest() for _, _, p in runs for x in p}), flush=True)
t0 = time.time()
jobs = [(arm, d, i, x) for arm, d, p in runs for i, x in enumerate(p)]
with ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(lambda j: ep(j[3]["code"], j[3].get("algorithm")), jobs))
print("evaluated in", round(time.time() - t0), "s", flush=True)
by_run = {}
for (arm, d, i, x), r in zip(jobs, res):
    by_run.setdefault((arm, d), []).append({"idx": i, "train": x["objective"], **r})
table = {}
for (arm, d), rows in by_run.items():
    json.dump(rows, open(os.path.join(d, "pop_endpoint.json"), "w", encoding="utf-8"), indent=1)
    c5 = [r["c500"] for r in rows]
    seed = d.split("_cs")[1][:5]
    table.setdefault(arm, []).append({"seed": seed, "n": len(rows), "med_c500": st.median(c5), "min_c500": min(c5),
                                      "frac_gt10": sum(v > 0.10 for v in c5) / len(c5), "med_c100": st.median(r["c100"] for r in rows),
                                      "train_best_c500": min(rows, key=lambda r: r["train"])["c500"]})
print(f"\n{'arm':4s} {'seed':6s} {'n':>2s} {'pop med c500%':>13s} {'pop min c500%':>13s} {'frac>10%':>9s} {'pop med c100%':>13s} {'best-train c500%':>16s}")
for arm in ARMS:
    for t in sorted(table.get(arm, []), key=lambda t: t["seed"]):
        print(f"{arm:4s} {t['seed']:6s} {t['n']:2d} {100*t['med_c500']:13.3f} {100*t['min_c500']:13.3f} {t['frac_gt10']:9.2f} {100*t['med_c100']:13.3f} {100*t['train_best_c500']:16.3f}")
print("\narm  runs  median-over-runs(pop med c500)  median(pop min c500)  mean frac>10%  median(pop med c100)")
for arm in ARMS:
    ts = table.get(arm, [])
    if ts:
        print(f"{arm:4s} {len(ts):4d} {100*st.median(t['med_c500'] for t in ts):28.3f} {100*st.median(t['min_c500'] for t in ts):20.3f} "
              f"{st.mean(t['frac_gt10'] for t in ts):14.3f} {100*st.median(t['med_c100'] for t in ts):20.3f}")
json.dump(table, open(os.path.join(ROOT, "pop_c500_table_B12_20260909.json"), "w", encoding="utf-8"), indent=1)
