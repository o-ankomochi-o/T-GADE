import os
"""Population-level C100/C500 of the generational final populations, same endpoint helper as
pop_c500.py; separate cache. Writes pop_endpoint.json per run and a per-arm table ($0, sandbox only)."""
import glob, hashlib, importlib.util, json, os, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

REPO = os.environ.get("TGADE_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/src")
spec = importlib.util.spec_from_file_location("run_v101_eoh_popg1", REPO + "/scripts/run_v101_eoh.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: E402

ROOT = os.environ.get("TGADE_RUNS", os.path.join(REPO, "results", "runs", "v101"))
ARMS = {"G1-fermion0": "g1_x2full_bh_sigE_fermion_T0_evt30", "G1-boson.003": "g1_x2full_bh_sigE_boson_T0.003_evt30",
        "G1-boson.03": "g1_x2full_bh_sigE_boson_T0.03_evt30"}
adapter = BpOnlineAdapter(train_k=5, items=5000, objective="train_c100", diversity_carrier="behaviour",
                          probe_len=64, eval_timeout=90, signature_source="probe", deterministic_only=False)
banks = {"c100": m._load_bank("confirmation_bank.json"), "c500": m._load_bank("confirmation_bank_c500.json")}
penalties = {k: m._bank_penalty(b) for k, b in banks.items()}
CACHE_PATH = os.path.join(ROOT, "pop_endpoint_cache_G1p_20260909.json")
cache = json.load(open(CACHE_PATH, encoding="utf-8")) if os.path.exists(CACHE_PATH) else {}
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
    for d in sorted(glob.glob(os.path.join(ROOT, f"{pre}_cs205*_2026*"))):
        rp = os.path.join(d, "r1", "tgade", "result_bp_online_seed101.json")
        if not os.path.exists(rp):
            continue
        pop = json.load(open(rp, encoding="utf-8")).get("population") or []
        pop = [x for x in pop if isinstance(x.get("energy"), (int, float)) and isinstance(x.get("genes"), dict) and x["genes"].get("code")]
        runs.append((arm, d, pop))
print("runs", len(runs), "individuals", sum(len(p) for _, _, p in runs), flush=True)
t0 = time.time()
jobs = [(arm, d, i, x) for arm, d, p in runs for i, x in enumerate(p)]
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(lambda j: ep(j[3]["genes"]["code"], j[3]["genes"].get("thought")), jobs))
print("evaluated in", round(time.time() - t0), "s", flush=True)
by_run = {}
for (arm, d, i, x), r in zip(jobs, res):
    by_run.setdefault((arm, d), []).append({"idx": i, "train": 2 * x["energy"], "energy": x["energy"], **r})  # train raw = 2E (E = min(raw,2)/2)
table = {}
for (arm, d), rows in by_run.items():
    json.dump(rows, open(os.path.join(d, "pop_endpoint.json"), "w", encoding="utf-8"), indent=1)
    c5 = [r["c500"] for r in rows]
    table.setdefault(arm, []).append({"seed": d.split("_cs")[1][:5], "n": len(rows), "med_c500": st.median(c5), "min_c500": min(c5),
                                      "frac_gt10": sum(v > 0.10 for v in c5) / len(c5), "frac_lt1": sum(v < 0.01 for v in c5) / len(c5)})
print("arm          runs  med(pop med c500)  med(pop min)  mean frac>10%  mean frac<1%")
for arm, ts in table.items():
    print(f"{arm:12s} {len(ts):4d} {100*st.median(t['med_c500'] for t in ts):17.3f} {100*st.median(t['min_c500'] for t in ts):13.3f} "
          f"{st.mean(t['frac_gt10'] for t in ts):14.3f} {st.mean(t['frac_lt1'] for t in ts):13.3f}")
json.dump(table, open(os.path.join(ROOT, "pop_c500_table_G1p_20260909.json"), "w", encoding="utf-8"), indent=1)
