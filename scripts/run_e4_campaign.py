"""Campaign runner with confirmation and transfer evaluation.
Computer-science machinery only: LLM-driven heuristic-code search with
evidence and budget engineering; evolutionary vocabulary is borrowed.

Paid mode is driven by ONE sealed FREEZE artifact (scripts/freeze_e4.py):
parameter-table blob (== git HEAD blob, path clean), E3 report digest, E3 commit,
code commit (bench paths clean), freeze message id. The E3 report is
re-validated by bench.e3_gate.validate_e3_report (recomputed from raw
evidence, fail closed). Parameters come only from the frozen table.

Per run-seed, in an ATTEMPT-specific directory s{seed}_a{k} (never reused):
  1. gen0 BANK (3N ceiling, stop at N valid; sealed gen0_bank.json)
  2./3. TGADE and MULTISTART in an order alternating by seed parity
  4. per-instance endpoints on the sealed c100/c500 banks; selected
     genotypes + endpoints sealed per arm
  Every attempt ends with a SEALED pair_record.json (PAIR_RETAINED,
  BANK_SHORT, ABORT_INFRA); non-retained outcomes consume a reserve seed.
A sealed immutable design_manifest.json is written before any call;
--resume requires exact equality with the recomputed design, re-verifies
every sealed record and referenced artifact from bytes, and starts a NEW
attempt directory for any seed without a sealed terminal record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grant_evo.bench import BenchConfig, BenchRun  # noqa: E402
from grant_evo.bench.budget import GlobalBudget  # noqa: E402
from grant_evo.bench.e4_freeze import FreezeError, verify_freeze  # noqa: E402
from grant_evo.bench.e4_params import MODEL, PREREG_PARAMS  # noqa: E402
from grant_evo.bench.multistart import (  # noqa: E402
    MultistartRun, bank_from_result, validate_multistart_evidence)
from grant_evo.bench.seal import check_seal as _sealed_ok  # noqa: E402
from grant_evo.bench.seal import write_seal  # noqa: E402
from grant_evo.tgade.engine import (  # noqa: E402
    PopulationExtinctionError, loci_canonical_digest)

REPO = Path(__file__).resolve().parents[1]
E4 = REPO / "experiments" / "e4"


class CampaignRefused(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic_json(path: Path, obj) -> str:
    body = json.dumps(obj, indent=1, default=str, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return _sha(body.encode("utf-8"))


def _seal(path: Path, sha: str) -> None:
    write_seal(path, sha)  # shared convention: <name>.sha256


def _load_bank(name: str) -> dict:
    raw = (E4 / name).read_bytes()
    sealed = dict(line.split(" sha256 ") for line in
                  (E4 / "bank_seal.sha256").read_text(encoding="utf-8").splitlines() if line)
    if _sha(raw) != sealed.get(name):
        raise CampaignRefused(f"{name} does not match its seal")
    bank = json.loads(raw)
    bank["_sha256"] = sealed[name]
    bank["_name"] = name
    return bank


def _bank_penalty(bank: dict) -> float:
    """ worst-case raw excess on THIS sealed bank."""
    inst = bank.get("instances") or {}
    if not inst:
        return 3.0  # offline mock banks carry no instances
    sys.path.insert(0, str(REPO / "third_party/EoH/examples/bp_online"))
    from get_instance import GetData
    gd = GetData()
    worst = []
    for v in inst.values():
        lb = float(gd.l1_bound(tuple(int(x) for x in v["items"]), int(v["capacity"])))
        worst.append((int(v["num_items"]) - lb) / lb)
    return float(sum(worst) / len(worst))


def _verify_freeze(freeze_path: Path) -> dict:
    """/P0-1: ONE shared verification (bench.e4_freeze)."""
    try:
        return verify_freeze(freeze_path, require_clean=True)
    except FreezeError as exc:
        raise CampaignRefused(str(exc)) from exc


def _endpoint(adapter, genes, banks: dict, penalties: dict) -> dict:
    out = {}
    for key, bank in banks.items():
        if genes is None:
            out[key] = {"raw": penalties[key], "penalty": True, "per_instance": None}
            continue
        per = adapter.endpoint_raw(genes, bank)
        if per is None:
            out[key] = {"raw": penalties[key], "penalty": True, "per_instance": None}
        else:
            vals = list(per.values()) if isinstance(per, dict) else [per]
            out[key] = {"raw": float(sum(vals) / len(vals)), "penalty": False,
                        "per_instance": per if isinstance(per, dict) else {"all": per}}
    return out


def _persist_selected(sdir: Path, arm: str, genes, adapter, endpoint: dict,
                      banks: dict, extra: dict) -> None:
    from grant_evo.bench.sandbox import image_digest
    rec = {"arm": arm, "genes": genes,
           "digest": (loci_canonical_digest(adapter.loci_view(genes))
                      if genes is not None else None),
           "endpoint": endpoint,
           "evaluator": {"sandbox_image": image_digest() if genes is not None else None,
                         "banks": {k: v.get("_sha256") for k, v in banks.items()}},
           **extra}
    _seal(sdir / f"selected_{arm}.json", _atomic_json(sdir / f"selected_{arm}.json", rec))


def run_attempt(seed: int, sdir: Path, *, adapter, client_factory, n: int, g: int,
                c_total: int, t_star: float, banks: dict, penalties: dict) -> dict:
    sdir.mkdir(parents=True, exist_ok=False)  # attempt directories are never reused
    rec: dict = {"seed": seed, "attempt_dir": sdir.name, "n": n, "generations": g,
                 "C": c_total, "t_star": t_star,
                 "arm_order": (["TGADE", "MULTISTART"] if seed % 2
                               else ["MULTISTART", "TGADE"]), "phases": {}}

    def finish(status: str) -> dict:
        rec["status"] = status
        _seal(sdir / "pair_record.json", _atomic_json(sdir / "pair_record.json", rec))
        return rec

    bank_client = client_factory("bank", seed)
    prod = MultistartRun(adapter, bank_client, fresh_calls=3 * n, stop_at_valid=n,
                         out_dir=str(sdir / "bank"), seed=seed, label="bank").run()
    validate_multistart_evidence(sdir / "bank", "bank", seed)
    valid = [(c["genes"], c) for c in prod["candidates"]
             if c["origin"] == "fresh" and c["energy"] is not None]
    rec["phases"]["bank"] = {"attempts": prod["attempts"], "valid": len(valid),
                             "ledger_sha256": prod["call_ledger_sha256"],
                             "spend_usd": getattr(bank_client, "spent_usd", None),
                             "attempt_log": getattr(bank_client, "attempt_log", None)}
    if len(valid) < n:
        return finish("BANK_SHORT")
    bank = bank_from_result(prod, [gt for gt, _ in valid[:n]])
    _seal(sdir / "bank" / "gen0_bank.json",
          _atomic_json(sdir / "bank" / "gen0_bank.json", dict(bank)))
    rec["phases"]["bank"]["sha256"] = bank["sha256"]

    def run_tgade() -> dict:
        arm: dict = {"name": "TGADE"}
        client = client_factory("tgade", seed)
        tdir = sdir / "tgade"
        cfg = BenchConfig(n=n, generations=g, temperature=t_star,
                          occupancy=PREREG_PARAMS["occupancy"],
                          strength=PREREG_PARAMS["strength"], seed=seed,
                          out_dir=str(tdir), gen0_bank_sha256=bank["sha256"],
                          gen0_bank_digests=tuple(bank["digests"]))
        genes = None
        try:
            res = BenchRun(adapter, cfg, client, gen0_bank=bank).run()
            certified = any(tdir.glob("_SUCCESS_*"))
            arm.update({"attempted_calls": res.manifest["llm_calls"],
                        "call_failures": res.manifest["llm_call_failures"],
                        "vacancies": [lg["vacancies"] for lg in res.generation_log],
                        "certified": certified})
            if certified:
                best = min(res.population, key=lambda i: i.energy)
                genes = best.genes
                arm.update({"train_energy": best.energy, "status": "OK"})
            else:
                arm["status"] = "ABORT_UNCERTIFIED"
        except PopulationExtinctionError as exc:
            arm.update({"status": "EXTINCT", "detail": str(exc)[:300], "certified": False})
        arm["spend_usd"] = getattr(client, "spent_usd", None)
        arm["attempt_log"] = getattr(client, "attempt_log", None)
        if arm["status"] in ("OK", "EXTINCT"):
            arm["endpoint"] = _endpoint(adapter, genes, banks, penalties)
            _persist_selected(sdir, "TGADE", genes, adapter, arm["endpoint"], banks,
                              {"status": arm["status"], "train_energy": arm.get("train_energy")})
        else:
            arm["endpoint"] = None
        return arm

    def run_multistart() -> dict:
        marm: dict = {"name": "MULTISTART"}
        client = client_factory("multistart", seed)
        mdir = sdir / "multistart"
        ms = MultistartRun(adapter, client, fresh_calls=c_total - n, bank=bank,
                           out_dir=str(mdir), seed=seed, label="ms").run()
        validate_multistart_evidence(mdir, "ms", seed, bank=bank,
                                     expected_fresh_calls=c_total - n)
        genes = None
        if ms["best_digest"] is not None:
            for c in ms["candidates"]:
                if c["digest"] == ms["best_digest"]:
                    genes = c.get("genes")
                    break
            if genes is None:
                for gt in bank["genotypes"]:
                    if loci_canonical_digest(adapter.loci_view(gt)) == ms["best_digest"]:
                        genes = gt
                        break
        marm.update({"train_energy": ms["best_energy"], "attempted_calls": ms["attempts"],
                     "status": "OK" if genes is not None else "NO_VALID",
                     "ledger_sha256": ms["call_ledger_sha256"],
                     "spend_usd": getattr(client, "spent_usd", None),
                     "attempt_log": getattr(client, "attempt_log", None)})
        marm["endpoint"] = _endpoint(adapter, genes, banks, penalties)
        _persist_selected(sdir, "MULTISTART", genes, adapter, marm["endpoint"], banks,
                          {"status": marm["status"], "train_energy": ms["best_energy"]})
        return marm

    arms: dict = {}
    for name in rec["arm_order"]:
        arms[name] = run_tgade() if name == "TGADE" else run_multistart()
    rec["arms"] = arms
    if arms["TGADE"]["status"] == "ABORT_UNCERTIFIED":
        return finish("ABORT_INFRA")
    return finish("PAIR_RETAINED")


def verify_attempt(sdir: Path, design: dict) -> dict:
    """ re-verify a sealed attempt from bytes before reuse."""
    pr = sdir / "pair_record.json"
    if not _sealed_ok(pr):
        raise CampaignRefused(f"{sdir.name}: pair record seal mismatch")
    rec = json.loads(pr.read_bytes())
    if rec.get("n") != design["n"] or rec.get("generations") != design["generations"] \
            or rec.get("C") != design["C"] or rec.get("t_star") != design["t_star"]:
        raise CampaignRefused(f"{sdir.name}: pair record design differs")
    if rec.get("status") == "PAIR_RETAINED":
        if not _sealed_ok(sdir / "bank" / "gen0_bank.json"):
            raise CampaignRefused(f"{sdir.name}: gen0 bank seal mismatch")
        for arm in ("TGADE", "MULTISTART"):
            sel = sdir / f"selected_{arm}.json"
            if not _sealed_ok(sel):
                raise CampaignRefused(f"{sdir.name}: selected_{arm} seal mismatch")
            s = json.loads(sel.read_bytes())
            if s.get("endpoint") != rec["arms"][arm].get("endpoint"):
                raise CampaignRefused(f"{sdir.name}: {arm} endpoint differs from its sealed artifact")
            if s.get("evaluator", {}).get("banks") != design["banks"]:
                raise CampaignRefused(f"{sdir.name}: {arm} evaluated on different banks")
        if rec["arms"]["TGADE"].get("certified") and not list((sdir / "tgade").glob("_SUCCESS_*")):
            raise CampaignRefused(f"{sdir.name}: TGADE certification marker missing")
        if not list((sdir / "multistart").glob("_MS_COMPLETE_*")):
            raise CampaignRefused(f"{sdir.name}: MULTISTART completion marker missing")
    return rec


def summarize(records: list[dict], banks: dict) -> dict:
    pairs = [r for r in records if r.get("status") == "PAIR_RETAINED"]
    summary: dict = {"n_pairs": len(pairs),
                     "replaced_seeds": [r["seed"] for r in records
                                        if r.get("status") in ("BANK_SHORT", "ABORT_INFRA")],
                     "per_bank": {}}
    for key in banks:
        tg = [r["arms"]["TGADE"]["endpoint"][key]["raw"] for r in pairs]
        ms = [r["arms"]["MULTISTART"]["endpoint"][key]["raw"] for r in pairs]
        diffs = [a - b for a, b in zip(tg, ms)]
        row = {"tgade": tg, "multistart": ms, "diff_tgade_minus_ms": diffs,
               "wins": sum(1 for d in diffs if d < 0),
               "ties": sum(1 for d in diffs if d == 0),
               "losses": sum(1 for d in diffs if d > 0),
               "tgade_penalties": sum(1 for r in pairs if r["arms"]["TGADE"]["endpoint"][key]["penalty"]),
               "ms_penalties": sum(1 for r in pairs if r["arms"]["MULTISTART"]["endpoint"][key]["penalty"])}
        nonzero = [d for d in diffs if d != 0]
        if nonzero:
            try:
                from scipy.stats import wilcoxon
                row["wilcoxon_one_sided_p"] = float(
                    wilcoxon(nonzero, alternative="less", zero_method="wilcox").pvalue)
            except Exception as exc:  # noqa: BLE001
                row["wilcoxon_error"] = f"{type(exc).__name__}: {exc}"
        summary["per_bank"][key] = row
    summary["claim_scope"] = ("EXPLORATORY pilot: no superiority claim; "
                              "c500 axis mandatory; train/c100-only claims forbidden")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--freeze", default=None, help="sealed freeze artifact (paid mode)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="offline only")
    ap.add_argument("--n", type=int, default=None, help="offline only")
    ap.add_argument("--generations", type=int, default=None, help="offline only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--offline-fail-bank-seeds", type=int, nargs="*", default=[],
                    help="offline only: force BANK_SHORT for these seeds (tests)")
    a = ap.parse_args()
    try:
        return _main(a)
    except CampaignRefused as exc:
        print(f"REFUSED: {exc}")
        return 2


def _main(a) -> int:
    if a.offline:
        n, g = a.n or 4, a.generations or 2
        seeds = a.seeds or [1, 2]
        reserve = [s + 100 for s in seeds]
        cap, t_star = 0.0, 0.5
        freeze = {"prereg_blob_sha": None, "e3_report_sha256": None, "code_commit": None,
                  "freeze_message_id": None}
        from grant_evo.bench.mock import MockProblem, ScriptableLLM
        adapter = MockProblem()
        banks = {"c100": {"capacity": 100, "instances": {}},
                 "c500": {"capacity": 500, "instances": {}}}
        fail_bank = set(a.offline_fail_bank_seeds)

        def client_factory(phase, seed):
            if phase == "bank" and seed in fail_bank:
                return ScriptableLLM(["fail"] * 100)
            return ScriptableLLM()
        authority = None
    else:
        if not (a.confirm_paid and a.freeze):
            raise CampaignRefused("paid campaign needs --confirm-paid and --freeze <artifact>")
        if any(v is not None for v in (a.seeds, a.n, a.generations)) or a.offline_fail_bank_seeds:
            raise CampaignRefused("paid mode takes its parameters from the frozen table only")
        freeze = _verify_freeze(Path(a.freeze))
        t_star = freeze["_t_star"]
        n, g = PREREG_PARAMS["n"], PREREG_PARAMS["generations"]
        seeds, reserve, cap = (PREREG_PARAMS["seeds"], PREREG_PARAMS["reserve_seeds"],
                               PREREG_PARAMS["cap_usd"])
        from grant_evo.bench.adapters.bp_online import BpOnlineAdapter
        from grant_evo.bench.clients import OpenRouterClient
        adapter = BpOnlineAdapter(train_k=5, items=5000)
        banks = {"c100": _load_bank("confirmation_bank.json"),
                 "c500": _load_bank("confirmation_bank_c500.json")}
        seed_base = {"bank": 10000, "tgade": 20000, "multistart": 30000}
        authority = None  # bound below to the campaign directory

        def client_factory(phase, seed):
            P = PREREG_PARAMS
            return OpenRouterClient(P["model"], temperature=P["t_sample"],
                                    budget_usd=cap,
                                    retries=P["transport_retries"],
                                    min_interval_s=P["min_interval_s"],
                                    max_tokens=P["max_tokens"],
                                    price_in_usd_per_m=P["price_in_usd_per_m"],
                                    price_out_usd_per_m=P["price_out_usd_per_m"],
                                    max_input_bytes=P["max_input_bytes"],
                                    chat_overhead_tokens=P["chat_overhead_tokens"],
                                    provider_order=P["provider_order"],
                                    allow_fallbacks=P["allow_fallbacks"],
                                    seed=seed_base[phase] + seed,
                                    global_budget=authority)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = Path(a.out or (REPO / "local" / "runs" / "e4" / (("offline_" if a.offline else "") + stamp)))
    c_total = n + 5 * n * g
    penalties = {k: _bank_penalty(b) for k, b in banks.items()}
    design = {"model": None if a.offline else MODEL, "n": n, "generations": g, "C": c_total,
              "seeds": seeds, "reserve_seeds": reserve, "t_star": t_star,
              "penalties_raw": penalties, "cap_usd": cap,
              "banks": {k: v.get("_sha256") for k, v in banks.items()},
              "prereg_blob_sha": freeze.get("prereg_blob_sha"),
              "e3_report_sha256": freeze.get("e3_report_sha256"),
              "code_commit": freeze.get("code_commit"),
              "freeze_id": freeze.get("freeze_message_id"), "offline": a.offline}
    dm = out / "design_manifest.json"
    if a.resume:
        if not _sealed_ok(dm):
            raise CampaignRefused("--resume: design manifest missing or seal mismatch")
        if json.loads(dm.read_bytes()) != json.loads(json.dumps(design, default=str, sort_keys=True)):
            raise CampaignRefused("--resume: design differs from the sealed manifest")
    else:
        out.mkdir(parents=True)
        _seal(dm, _atomic_json(dm, design))
    if not a.offline:  # client_factory closes over this enclosing variable
        authority = GlobalBudget(out / "budget_authority.json", cap_usd=cap)

    manifest = {"design": design, "records": []}
    queue = list(seeds)
    reserve_q = list(reserve)
    target_pairs = len(seeds)
    retained = 0
    while queue:
        seed = queue.pop(0)
        attempts = sorted(out.glob(f"s{seed}_a*"), key=lambda p: int(p.name.split("_a")[1]))
        rec = None
        if attempts:
            last = attempts[-1]
            if (last / "pair_record.json").exists():
                if not a.resume:
                    raise CampaignRefused(f"{last.name} exists; use --resume")
                rec = verify_attempt(last, design)  # sealed terminal record: replay
        if rec is None:
            k = (int(attempts[-1].name.split("_a")[1]) + 1) if attempts else 1
            rec = run_attempt(seed, out / f"s{seed}_a{k}", adapter=adapter,
                              client_factory=client_factory, n=n, g=g, c_total=c_total,
                              t_star=t_star, banks=banks, penalties=penalties)
        manifest["records"].append(rec)
        if rec.get("status") == "PAIR_RETAINED":
            retained += 1
        elif reserve_q:
            queue.append(reserve_q.pop(0))
        _atomic_json(out / "campaign_progress.json", manifest)
    manifest["summary"] = summarize(manifest["records"], banks)
    manifest["summary"]["fixed_n_target"] = target_pairs
    manifest["summary"]["fixed_n_met"] = retained >= target_pairs
    _seal(out / "campaign_summary.json", _atomic_json(out / "campaign_summary.json", manifest))
    print(json.dumps({"out": str(out), "pairs": retained, "target": target_pairs,
                      "c100_wins": manifest["summary"]["per_bank"]["c100"].get("wins"),
                      "c500_wins": manifest["summary"]["per_bank"]["c500"].get("wins")}))
    return 0 if retained >= target_pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
