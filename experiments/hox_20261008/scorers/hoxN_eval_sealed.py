"""Sealed held-out readout of the steady-state cohorts (EoH reference and host-equipped T-GADE) at call budgets.

usage (from the repository root, after `pip install -r requirements.txt`):
  python experiments/hox_20261008/scorers/hoxN_eval_sealed.py --bank experiments/hox_20261008/sealed_bank/sealed_c100.json \
      --seal experiments/hox_20261008/sealed_bank/seal.sha256 --out OUT.json COHORT_DIR [COHORT_DIR ...]
COHORT_DIR holds one run directory per arm and seed, named <cohort>_bp_<date>_<time>_<arm>_cs<seed>ORT_DIR [COHORT_DIR ...]
Selection at budget b: among registrations whose total calls (operator + host calls STARTED before the registration) <= b,
the lowest training objective (5 decimals); ties -> earliest registration; the gen-0 population counts at 0 calls. 'end'
applies the same rule to all registrations. A run that has not reached b is missing at b. The selected code is scored with
the adapter's endpoint_raw on every instance of the bank (mean raw excess over the L1 lower bound); a failed evaluation is
replaced by the bank's worst case and flagged. Codes are cached by sha256. Pairs share the client seed; exact two-sided
Wilcoxon signed-rank test (zero differences dropped), as in hox2_readout.py.
"""
import argparse, bisect, hashlib, itertools, json, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
for p in ("src", "third_party/EoH/eoh/src", "third_party/EoH/examples/bp_online"):
    sys.path.insert(0, str(ROOT / p))
LABEL = re.compile(r"^hox[0-9a-z]+_bp_[0-9]{8}_[0-9]{4}_(eoh3200|eoh800|h_improve_t0003|h_improve_T[0-9.]+|h_improve)_cs([0-9]+)$")
PAIRS = ()  # filled at runtime from the arms present


def rows(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if p.exists() else []


def wilcoxon_exact(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    r, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    tot = sum(r); w = sum(r[i] for i in range(n) if d[i] > 0); obs = min(w, tot - w)
    hit = sum(min(x, tot - x) <= obs + 1e-9 for x in (sum(r[i] for i in range(n) if s[i]) for s in itertools.product((0, 1), repeat=n)))
    return hit / 2 ** n


def select(run_dir, budgets):
    """{budget: (registration index, objective, code)} under the pre-registered rule; None if the run has not reached b."""
    starts = sorted([r["ts"] for r in rows(run_dir / "call_ledger_eoh.jsonl") if r.get("event") == "call_started"]
                    + [r["ts"] for r in rows(run_dir / "call_ledger_repair.jsonl") if r.get("event") == "call_started"])
    cands = []  # (calls consumed, objective, registration order, code)
    for k, r in enumerate(rows(run_dir / "registration.jsonl")):
        for x in (r.get("pop_in_full") or []):
            if x.get("objective") is not None:
                cands.append((0, x["objective"], -1, x["code"]))
        nc = r.get("newcomer")
        if nc and nc.get("objective") is not None:
            cands.append((bisect.bisect_right(starts, r["ts"]), nc["objective"], k, nc["code"]))
    total = len(starts)
    out = {}
    for b in budgets:
        lim = total if b == "end" else int(b)
        if b != "end" and lim > total:
            out[b] = None
            continue
        pool = [c for c in cands if c[0] <= lim]
        out[b] = min(pool, key=lambda c: (c[1], c[2])) if pool else None
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohorts", nargs="+")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--budgets", default="200,400,800,1600,end")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = Path(a.bank).read_bytes()
    sealed = dict(line.split(" sha256 ") for line in Path(a.seal).read_text(encoding="utf-8").splitlines() if line)
    if sealed.get(Path(a.bank).name) != hashlib.sha256(raw).hexdigest():
        sys.exit("REFUSED: bank does not match its seal")
    bank = json.loads(raw)
    from get_instance import GetData  # noqa: PLC0415
    from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: PLC0415
    gd = GetData()
    worst = statistics.mean((int(v["num_items"]) - (lb := float(gd.l1_bound(tuple(int(x) for x in v["items"]), int(v["capacity"]))))) / lb
                            for v in bank["instances"].values())
    adapter = BpOnlineAdapter(train_k=5, items=5000, eval_timeout=30.0, deterministic_only=True)
    budgets = [b if b == "end" else int(b) for b in a.budgets.split(",")]
    cache, runs = {}, []
    for c in a.cohorts:
        for d in sorted(Path(c).iterdir()):
            m = LABEL.match(d.name)
            if not (d.is_dir() and m):
                continue
            sel, total = select(d, budgets)
            rec = {"label": d.name, "arm": m.group(1), "seed": int(m.group(2)), "calls_total": total, "done": (d / "summary.json").exists(), "at": {}}
            for b in budgets:
                s = sel[b]
                if s is None:
                    rec["at"][str(b)] = None
                    continue
                sha = hashlib.sha256(s[3].encode("utf-8")).hexdigest()
                if sha not in cache:
                    per = adapter.endpoint_raw({"code": s[3]}, bank)
                    cache[sha] = (worst, True) if per is None else (statistics.mean(per.values()), False)
                val, failed = cache[sha]
                rec["at"][str(b)] = {"calls": s[0], "train": s[1], "registration": s[2], "code_sha256": sha, "value": val, "failed": failed}
            runs.append(rec)
            print(rec["label"], {b: (None if v is None else round(100 * v["value"], 3)) for b, v in rec["at"].items()}, flush=True)
    by = {(r["arm"], r["seed"]): r for r in runs}
    tests = []
    arms = sorted({r["arm"] for r in runs})
    for x, y in [(x, y) for x in arms for y in arms if x != y]:
        for b in budgets:
            pr = [(by[(x, s)]["at"][str(b)]["value"], by[(y, s)]["at"][str(b)]["value"], s) for (arm, s) in by if arm == x and (y, s) in by
                  and by[(x, s)]["at"][str(b)] is not None and by[(y, s)]["at"][str(b)] is not None]
            if pr:
                diff = [p - q for p, q, _ in pr]
                tests.append({"pair": f"{x} vs {y}", "budget": str(b), "n": len(pr), "wins": sum(v < 0 for v in diff), "losses": sum(v > 0 for v in diff),
                              "median_diff_pct": round(100 * statistics.median(diff), 4), "p": wilcoxon_exact(diff),
                              "median_x_pct": round(100 * statistics.median(p for p, _, _ in pr), 4), "median_y_pct": round(100 * statistics.median(q for _, q, _ in pr), 4)})
    for t in tests:
        print(t)
    Path(a.out).write_text(json.dumps({"bank": Path(a.bank).name, "bank_sha256": sealed[Path(a.bank).name], "worst_case": worst,
                                       "budgets": [str(b) for b in budgets], "runs": runs, "tests": tests}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    assert wilcoxon_exact([1, 2, 3, 4, 5]) == 0.0625
    main()
