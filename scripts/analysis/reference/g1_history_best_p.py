import os
"""G1 reconciliation (PM review 2026-09-08 §1): recompute the HISTORY-best artifact of each finished G1 run
from result lineage (all evaluated candidates with energy), evaluate it on the C100/C500 banks with the
same endpoint helper as the steady-state runner, and store g1_history_best.json per run. $0 (sandbox only)."""
import glob, hashlib, importlib.util, json, os, sys, time

REPO = os.environ.get("TGADE_REPO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/src")
spec = importlib.util.spec_from_file_location("run_v101_eoh_g1hb", REPO + "/scripts/run_v101_eoh.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: E402

ROOT = os.environ.get("TGADE_RUNS", os.path.join(REPO, "results", "runs", "v101"))
adapter = BpOnlineAdapter(train_k=5, items=5000, objective="train_c100", diversity_carrier="behaviour",
                          probe_len=64, eval_timeout=90, signature_source="probe", deterministic_only=False)
banks = {"c100": m._load_bank("confirmation_bank.json"), "c500": m._load_bank("confirmation_bank_c500.json")}
penalties = {k: m._bank_penalty(b) for k, b in banks.items()}
CACHE = os.path.join(ROOT, "pop_endpoint_cache_G1hb_20260908.json")
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "g1_x2full_bh_sigE_*_evt30_cs205*_2026*"))):
    rp = os.path.join(d, "r1", "tgade", "result_bp_online_seed101.json")
    out = os.path.join(d, "g1_history_best.json")
    if not os.path.exists(rp):
        continue
    if os.path.exists(out):
        rows.append(json.load(open(out, encoding="utf-8"))); continue
    o = json.load(open(rp, encoding="utf-8"))
    lin = [x for x in o.get("lineage", []) if isinstance(x.get("energy"), (int, float)) and x.get("genes")]
    if not lin:
        continue
    best = min(lin, key=lambda x: x["energy"])
    genes = best["genes"]; code = genes.get("code") if isinstance(genes, dict) else None
    if not code:
        continue
    key = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if key not in cache:
        r = m._endpoint(adapter, {"code": code, "thought": genes.get("thought", "")}, banks, penalties)
        cache[key] = {"c100": r["c100"]["raw"], "c500": r["c500"]["raw"], "penalty": r["c500"]["penalty"] or r["c100"]["penalty"],
                      "c100_inst": r["c100"]["per_instance"], "c500_inst": r["c500"]["per_instance"]}
        json.dump(cache, open(CACHE, "w", encoding="utf-8"))
    fin = o.get("population") or []
    fin_e = [x.get("energy") for x in fin if isinstance(x.get("energy"), (int, float))]
    rec = {"run": os.path.basename(d), "history_best_energy": best["energy"], "history_best_gen": best.get("gen"),
           "history_best_id": best.get("id"), "n_evaluated": len(lin), "final_best_energy": min(fin_e) if fin_e else None,
           "history_best_c100": cache[key]["c100"], "history_best_c500": cache[key]["c500"], "code_sha256": key}
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
    rows.append(rec)
    print(rec["run"][:60], "hist E", round(best["energy"], 5), "gen", best.get("gen"), "final E", None if not fin_e else round(min(fin_e), 5),
          "c100 %.3f c500 %.3f" % (100 * rec["history_best_c100"], 100 * rec["history_best_c500"]), flush=True)
print("done", len(rows), time.strftime("%H:%M:%S"))
