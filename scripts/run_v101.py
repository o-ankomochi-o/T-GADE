"""V101 single-arm runner (seed-101 verification ladder, V2/V3).

Runs ONE T-GADE arm from the SEALED seed-101 gen0 bank with explicit
BenchConfig overrides (temperature, occupancy, mutate_children,
integrity_mode, strength) so each ladder step is one declared difference.
Computer-science machinery only (LLM program evolution on online bin packing).

Paid runs need --confirm-paid; every run has its OWN hard cap (no shared
cross-process ledger: item 4). Outputs (per repeat):
  <out>/tgade/            engine evidence (result, ledger, responses, markers)
  <out>/summary.json      train best, c100/c500 endpoints, per-generation
                          selection audit, per-operator child statistics,
                          spend by role; sealed with summary.json.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grant_evo.bench.e4_params import PREREG_PARAMS  # noqa: E402
from grant_evo.bench.seal import check_seal, sha256_bytes, write_seal  # noqa: E402
from grant_evo.tgade.engine import BenchConfig, BenchRun, loci_canonical_digest  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_e4_campaign", REPO / "scripts" / "run_e4_campaign.py")
_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e4)  # main() is guarded; we reuse its helpers
_load_bank, _endpoint, _bank_penalty, _atomic_json = (_e4._load_bank, _e4._endpoint,
                                                      _e4._bank_penalty, _e4._atomic_json)

DEFAULT_BANK = REPO / "data/gen0_bank.json"


def _sandbox_version() -> dict:
    from grant_evo.bench import sandbox as sb  # noqa: PLC0415
    return {"protocol": sb.PROTOCOL, "image": sb.image_digest()}


def _git_head() -> "str | None":
    import subprocess  # noqa: PLC0415
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def _load_gen0_bank(path: Path) -> dict:
    if not check_seal(path):  # refuses a bank whose seal does not match
        raise SystemExit(f"REFUSED: {path} does not match its seal")
    return json.loads(path.read_bytes())


def _op_stats(res) -> dict:
    """Per-operator child statistics from the lineage (E = raw excess / 2)."""
    L = {i.id: i for i in res.lineage}
    out: dict = {}
    for i in res.lineage:
        if i.gen == 0 or i.op in ("survivor", "elite_carryover", "init", "init_bank", "clone"):
            continue
        pe = [L[p].energy for p in i.parent_ids if p in L and L[p].energy is not None]
        if not pe:
            continue
        e0 = min(pe)
        k = out.setdefault(i.op, {"n": 0, "viable": 0, "improve": 0, "degrade": 0,
                                   "good_parent_n": 0, "good_parent_degrade": 0, "best_child": None})
        k["n"] += 1
        if i.energy is None:
            continue
        k["viable"] += 1
        k["improve"] += int(i.energy < e0)
        k["degrade"] += int(i.energy > e0)
        if e0 < 0.05:
            k["good_parent_n"] += 1
            k["good_parent_degrade"] += int(i.energy > e0)
        k["best_child"] = i.energy if k["best_child"] is None else min(k["best_child"], i.energy)
    return out


def _selection_audit(res) -> list:
    L = {i.id: i for i in res.lineage}
    rows = []
    best = 1.0
    for g in res.generation_log:
        pool = [p for p in g["pool_ids"] if p in L and L[p].energy is not None]
        if not pool:
            continue
        E = {p: L[p].energy for p in pool}
        ranked = sorted(E, key=E.get)
        sel = list(g["selected_ids"])
        top = set(ranked[:len(sel)])
        se = [E[s] for s in sel if s in E]
        best = min(best, g["best_energy"])
        rows.append({"gen": g["gen"], "pool_valid": len(pool), "best_so_far": best,
                     "pool_best": min(E.values()), "top_k_mean": sum(E[x] for x in ranked[:len(sel)]) / len(sel),
                     "selected_mean": sum(se) / len(se), "selected_worst": max(se),
                     "top_k_survived": len(top & set(sel)), "clones": len(sel) - len(set(sel)),
                     "garbage_selected": sum(1 for x in se if x >= 0.1)})
    return rows


def _spend_by_role(res) -> dict:
    out: dict = {}
    for c in res.call_ledger:
        r = out.setdefault(getattr(c, "role", None) or "?", {"calls": 0, "prompt_tokens": 0,
                                                             "completion_tokens": 0, "usd": 0.0})
        r["calls"] += 1
        u = c.usage or {}
        r["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        r["completion_tokens"] += int(u.get("completion_tokens") or 0)
        r["usd"] += float(u.get("charged_usd") or u.get("cost_usd") or 0.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--offline", action="store_true", help="mock adapter + scripted LLM ($0 plumbing check)")
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--client-seed", type=int, default=None, help="default 20000+seed (= E4 TGADE arm)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--n", type=int, default=PREREG_PARAMS["n"])
    ap.add_argument("--generations", type=int, default=PREREG_PARAMS["generations"])
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--occupancy", default=PREREG_PARAMS["occupancy"], choices=["boson", "fermion"])
    ap.add_argument("--strength", default=PREREG_PARAMS["strength"], choices=["weak", "mid", "strong"])
    ap.add_argument("--eval-timeout", type=float, default=None,
                    help="lethal-candidate wall-clock cap (s) of one train evaluation / signature probe; default 120 (historical)")
    ap.add_argument("--signature-source", default="probe", choices=["probe", "energy"],
                    help="'energy' = behaviour signature as a by-product of the energy evaluation (all steps of train instance 0)")
    ap.add_argument("--deterministic-only", action="store_true",
                    help="determinism gate: candidates using randomness APIs are invalid")
    ap.add_argument("--parent-alloc", default="uniform", choices=["uniform", "rank"],
                    help="EoH rank allocation of mutation targets and crossover pairs")
    ap.add_argument("--operator-policy", default="tgade", choices=["tgade", "eoh"],
                    help="survivors kept unchanged + N+1 EoH-style offspring")
    ap.add_argument("--child-post-ops", default="mutate+integrity",
                    choices=["mutate+integrity", "integrity", "none"],
                    help="post-crossover stack on children (V101 D2 ablations naming)")
    ap.add_argument("--repeat-mode", default="paired", choices=["same", "paired"],
                    help="same: every repeat reuses the client and solver seeds (same-request "
                         "reproducibility); paired: repeat r uses client seed +1000(r-1) and "
                         "solver seed +10(r-1) (independent repeats; pair the same offsets across arms)")
    ap.add_argument("--integrity", default="full", choices=["full", "skip"])
    ap.add_argument("--integrity-kind", default="syntax", choices=["syntax", "align_thought", "align_thought_meas", "align_code"],
                    help="Step-6 integrity repair: legacy syntax-only, or description<->code alignment")
    ap.add_argument("--gen0-integrity", action="store_true", help="apply Step-6 repair to the gen0 bank too")
    ap.add_argument("--parent-policy", default="mutate_all",
                    choices=["mutate_all", "keep_originals_mutate_duplicates"],
                    help="variant: keep survivor originals, mutate only bosonic duplicates")
    ap.add_argument("--model", default=None, help="operator-model override (affinity study); requires --price-in/--price-out; provider fallbacks allowed")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider for the override model (no fallbacks); default: fallbacks allowed")
    ap.add_argument("--objective", default="train_c100", choices=["train_c100", "regime_max"],
                    help="energy: canonical C=100 train set, or robust two-capacity Q=max(mean excess C100 bank, C500 bank)")
    ap.add_argument("--diversity-carrier", default="behaviour", choices=["behaviour", "nl", "hybrid"],
                    help="log det carrier: sandbox behaviour signature (canonical) or fixed-model NL embedding of the thought")
    ap.add_argument("--mutation-template", default="m1", choices=["m1", "m2", "mix"],
                    help="official EoH mutation template used by Step 5 (V101 D3: m2 = parameter adjustment)")
    ap.add_argument("--eval-workers", type=int, default=16)
    ap.add_argument("--hybrid-weights", default="0.8,0.1,0.1", help="hybrid carrier block weights bh,desc,code (sum 1)")
    ap.add_argument("--eval-seed", type=int, default=None, help="evaluation-seed panel: seed candidate RNG inside the sandbox; default None = historical evaluator")
    ap.add_argument("--probe-len", type=int, default=64, help="behaviour-signature probe length (items of TRAIN instance 0); V3 variation uses 512")
    ap.add_argument("--llm-workers", type=int, default=0, help="0 = legacy sequential operator calls; >=1 = batched parallel path (rng per_op_v2)")
    ap.add_argument("--cap-usd", type=float, default=0.5, help="hard cap PER RUN (own ledger)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not a.offline and not a.confirm_paid:
        print("REFUSED: paid run needs --confirm-paid (or use --offline)")
        return 2

    stamp = time.strftime("%Y%m%dT%H%M%S")
    root = Path(a.out or (REPO / "runs" / (("offline_" if a.offline else "") + a.label + "_" + stamp)))
    root.mkdir(parents=True, exist_ok=False)

    if a.offline:
        from grant_evo.bench.mock import MockProblem, ScriptableLLM  # noqa: PLC0415
        adapter = MockProblem()
        banks = {"c100": {"capacity": 100, "instances": {}}, "c500": {"capacity": 500, "instances": {}}}
        bank = None
        cseed = None
        client_factory = lambda client_seed: ScriptableLLM()  # noqa: E731
    else:
        sys.path.insert(0, str(REPO / "third_party/EoH/examples/bp_online"))
        from grant_evo.bench.adapters.bp_online import BpOnlineAdapter  # noqa: PLC0415
        from grant_evo.bench.clients import OpenRouterClient  # noqa: PLC0415
        adapter = BpOnlineAdapter(train_k=5, items=5000, mutation_template=a.mutation_template,
                                  diversity_carrier=a.diversity_carrier, objective=a.objective,
                                  integrity_kind=a.integrity_kind, eval_timeout=a.eval_timeout,
                                  signature_source=a.signature_source, deterministic_only=a.deterministic_only,
                                  hybrid_weights=tuple(float(x) for x in a.hybrid_weights.split(",")), probe_len=a.probe_len, eval_seed=a.eval_seed)
        banks = {"c100": _load_bank("confirmation_bank.json"), "c500": _load_bank("confirmation_bank_c500.json")}
        bank = _load_gen0_bank(Path(a.bank))
        P = PREREG_PARAMS
        if a.model and (a.price_in is None or a.price_out is None):
            raise SystemExit("REFUSED: --model needs --price-in and --price-out")
        OP_MODEL = a.model or P["model"]
        OP_PIN = a.price_in if a.model else P["price_in_usd_per_m"]
        OP_POUT = a.price_out if a.model else P["price_out_usd_per_m"]
        OP_PROV = ([a.provider] if a.provider else None) if a.model else P["provider_order"]
        OP_FB = (not a.provider) if a.model else P["allow_fallbacks"]
        cseed = a.client_seed if a.client_seed is not None else 20000 + a.seed

        def client_factory(client_seed):
            return OpenRouterClient(OP_MODEL, temperature=P["t_sample"], budget_usd=a.cap_usd,
                                    retries=P["transport_retries"], min_interval_s=P["min_interval_s"],
                                    max_tokens=P["max_tokens"], price_in_usd_per_m=OP_PIN,
                                    price_out_usd_per_m=OP_POUT,
                                    max_input_bytes=P["max_input_bytes"],
                                    chat_overhead_tokens=P["chat_overhead_tokens"],
                                    provider_order=OP_PROV, allow_fallbacks=OP_FB,
                                    seed=client_seed)
    penalties = {k: _bank_penalty(b) for k, b in banks.items()}

    design = {"label": a.label, "offline": a.offline, "seed": a.seed,
              "client_seed": None if a.offline else cseed, "bank": None if bank is None else bank["sha256"],
              "n": a.n, "generations": a.generations, "temperature": a.temperature,
              "occupancy": a.occupancy, "strength": a.strength,
              "occupancy_rule": ("fermion: one individual per genotype" if a.occupancy == "fermion" else "boson: unlimited, copies allowed"),
              "genotype_key": "sha256 of the canonical JSON of the loci view {thought, code} (exact match)",
              "parent_alloc": a.parent_alloc, "operator_policy": a.operator_policy,
              "eval_timeout_s": (a.eval_timeout if a.eval_timeout is not None else 120.0),
              "signature_source": a.signature_source, "deterministic_only": a.deterministic_only,
              "child_post_ops": a.child_post_ops, "integrity_mode": a.integrity, "gen0_integrity": a.gen0_integrity, "llm_workers": a.llm_workers, "rng_schedule_version": ("per_op_v2" if a.llm_workers >= 1 else "shared_v1"),
              "parent_policy": a.parent_policy, "mutation_template": a.mutation_template, "diversity_carrier": a.diversity_carrier, "hybrid_weights": a.hybrid_weights, "probe_len": a.probe_len, "eval_seed": a.eval_seed, "objective": a.objective, "integrity_kind": a.integrity_kind,
              "operator_spec": getattr(adapter, "operator_spec", None),
              "eval_workers": a.eval_workers, "cap_usd_per_run": a.cap_usd, "repeat": a.repeat,
              "repeat_mode": a.repeat_mode,
              "repeat_seeds": [{"repeat": r, "solver_seed": a.seed + (10 * (r - 1) if a.repeat_mode == "paired" else 0),
                                "client_seed": None if cseed is None else cseed + (1000 * (r - 1) if a.repeat_mode == "paired" else 0)}
                               for r in range(1, a.repeat + 1)],
              "model": None if a.offline else OP_MODEL, "operator_model_override": bool(a.model),
              "execution": None if a.offline else {k: PREREG_PARAMS[k] for k in
                                                   ("provider_order", "allow_fallbacks", "min_interval_s",
                                                    "transport_retries", "max_tokens", "t_sample")},
              "banks": {k: v.get("_sha256") for k, v in banks.items()}, "penalties_raw": penalties,
              "sandbox_protocol": _sandbox_version(), "code_commit": _git_head()}
    write_seal(root / "design.json", _atomic_json(root / "design.json", design))

    for r in range(1, a.repeat + 1):
        rdir = root / f"r{r}"
        tdir = rdir / "tgade"
        rdir.mkdir()
        rs = design["repeat_seeds"][r - 1]
        cfg_kwargs = dict(n=a.n, generations=a.generations, temperature=a.temperature,
                          occupancy=a.occupancy, strength=a.strength, seed=rs["solver_seed"],
                          child_post_ops=a.child_post_ops, integrity_mode=a.integrity, gen0_integrity=a.gen0_integrity, llm_workers=a.llm_workers, rng_schedule_version=("per_op_v2" if a.llm_workers >= 1 else "shared_v1"),
                          parent_policy=a.parent_policy, parent_alloc=a.parent_alloc,
                          operator_policy=a.operator_policy, out_dir=str(tdir))
        if bank is not None:
            cfg_kwargs.update(gen0_bank_sha256=bank["sha256"], gen0_bank_digests=tuple(bank["digests"]))
        cfg = BenchConfig(**cfg_kwargs)
        t0 = time.time()
        res = BenchRun(adapter, cfg, client_factory(rs["client_seed"]), gen0_bank=bank,
                       eval_workers=a.eval_workers).run()
        elapsed = time.time() - t0
        markers = sorted(p.name for p in tdir.iterdir() if p.name.startswith("_"))
        certified = any(m.startswith("_SUCCESS") for m in markers)
        best = min(res.population, key=lambda i: i.energy) if res.population else None
        genes = best.genes if best is not None else None
        endpoint = _endpoint(adapter, genes, banks, penalties)
        summary = {"repeat": r, "label": a.label, "markers": markers, "certified": certified,
                   "elapsed_s": round(elapsed, 1), "llm_calls": res.manifest.get("llm_calls"),
                   "llm_call_failures": res.manifest.get("llm_call_failures"),
                   "usage_totals": res.manifest.get("llm_usage_totals"),
                   "train_energy_best": None if best is None else best.energy,
                   "train_raw_excess_best": None if best is None else best.energy * 2.0,
                   "best_digest": None if genes is None else loci_canonical_digest(adapter.loci_view(genes)),
                   "endpoint": endpoint,
                   "selection_audit": _selection_audit(res), "op_stats": _op_stats(res),
                   "spend_by_role": _spend_by_role(res),
                   "final_population_energies": sorted(i.energy for i in res.population)}
        sha = _atomic_json(rdir / "summary.json", summary)
        write_seal(rdir / "summary.json", sha)
        if genes is not None:
            write_seal(rdir / "selected.json", _atomic_json(rdir / "selected.json",
                       {"genes": genes, "digest": summary["best_digest"], "endpoint": endpoint}))
        print(json.dumps({"repeat": r, "certified": certified, "markers": markers,
                          "train_raw_excess_best": summary["train_raw_excess_best"],
                          "c100": endpoint["c100"]["raw"], "c500": endpoint["c500"]["raw"],
                          "calls": summary["llm_calls"], "elapsed_s": summary["elapsed_s"],
                          "usd": round(sum(v["usd"] for v in summary["spend_by_role"].values()), 4)}))
    print("DONE", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
