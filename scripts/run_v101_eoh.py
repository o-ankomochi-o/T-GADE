"""V101 EoH reference arm: the commit-pinned EoH implementation
(third_party/EoH, upstream 4725457, package unmodified) connected to the
COMMON environment of the T-GADE arm: the same sealed seed-101 gen0 bank,
LLM, provider pinning, ledgered client and sandbox evaluator. It is a
reference arm, not a reproduction of the 2024 paper run.

Injection / wrapper points (all outside the EoH package, all declared in
design.json):
  1. Evolution.llm  -> LLMShim(get_response): ledgered OpenRouterClient with a
     HARD LOGICAL-CALL CAP (EoH may spend several calls per sample: extraction
     retries x duplicate retries); arms are matched on logical calls.
  2. evolution._eval_with_timeout / eoh._eval_with_timeout -> sandboxed
     evaluate_program (raw mean excess); the official path executes candidate
     code on the host, which the sandbox discipline forbids.
  3. Evolution.evaluate_seeds -> output sorted by objective (multiset
     unchanged) and the best tracker seeded, because upstream keeps the seed
     file order and ranks parents by list position.
  4. Runtime side effect inherited from BpOnlineAdapter._lazy_imports():
     evolution.InterfaceLLM is stubbed so Evolution() does not ping a network
     client at construction; the stub is then replaced by the shim.
Computer-science machinery only (LLM program evolution on online bin packing).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party/EoH/eoh/src"))
sys.path.insert(0, str(REPO / "third_party/EoH/examples/bp_online"))

from grant_evo.bench.e4_params import PREREG_PARAMS  # noqa: E402
from grant_evo.bench.seal import check_seal, write_seal  # noqa: E402
from grant_evo.tgade.engine import LLMResult  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_e4_campaign", REPO / "scripts" / "run_e4_campaign.py")
_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e4)
_load_bank, _endpoint, _bank_penalty, _atomic_json = (_e4._load_bank, _e4._endpoint,
                                                      _e4._bank_penalty, _e4._atomic_json)
DEFAULT_BANK = REPO / "data/gen0_bank.json"
NO_THINK = "/no_think\n"  # declared wrapper, identical to the T-GADE adapter


def _git_head(cwd: Path) -> "str | None":
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


class LLMShim:
    """EoH LLM interface (get_response) over the ledgered client.

    Durable started -> terminal rows per LOGICAL call (like the engine),
    usage/attempts persisted, cost of refused-after-cap calls is zero
    (never counted as paid failures), exception cost taken from
    BudgetExceeded.usage."""

    def __init__(self, client, max_calls: int, out: Path, no_think: bool, version=None):
        self.client, self.max_calls, self.no_think = client, max_calls, no_think
        self.calls = self.refused = self.failures = self.physical_attempts = 0
        self._lock = threading.Lock()  # P1: cap/seq/ledger atomic under samplers>1
        self.version = version  # callable -> population version (registration count) seen by this call's sampler
        self.tls = threading.local()  # per sampler thread: seq / pop_version of its latest call (read at registration)
        self.usd = 0.0
        self.prompt_tokens = self.completion_tokens = 0
        (out / "responses").mkdir(exist_ok=True)
        self._resp_dir = out / "responses"
        self._fh = open(out / "call_ledger_eoh.jsonl", "w", encoding="utf-8", newline="\n")

    def _durable(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        import os  # noqa: PLC0415
        os.fsync(self._fh.fileno())  # engine-equivalent durability

    def _account(self, usage: dict) -> None:
        self.usd += float(usage.get("charged_usd") or usage.get("cost_usd") or 0.0)
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        att = usage.get("attempts")
        # attempts list is authoritative when present (an empty list = the cap

        # T80 §1); only a client that reports no attempts at all counts as 1.
        self.physical_attempts += len(att) if isinstance(att, list) else 1

    def get_response(self, prompt_content: str) -> str:
        prompt = (NO_THINK if self.no_think else "") + prompt_content
        psha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._lock:  # cap check + seq + started row are one atomic step
            if self.calls >= self.max_calls:
                self.refused += 1
                self._durable({"event": "call_refused", "ts": time.time(), "reason": "logical call cap",
                               "refused_index": self.refused})
                return ""  # budget exhausted: EoH records a failed generation, no cost
            self.calls += 1
            seq = self.calls
            pv = self.version() if self.version is not None else None
            self._durable({"event": "call_started", "ts": time.time(), "seq": seq, "prompt_sha256": psha,
                           "pop_version": pv, "thread": threading.current_thread().name})
        self.tls.seq, self.tls.pop_version = seq, pv
        call_id = f"eoh-{seq}"  # distinct provider seed per logical call (the client derives its seed from call_id)
        t0 = time.time()
        try:
            res = self.client(prompt, call_id=call_id)
        except Exception as exc:  # BudgetExceeded etc.; transport retries already happened inside
            usage = dict(getattr(exc, "usage", None) or {})
            with self._lock:
                self.failures += 1
                self._account(usage)
                self._durable({"event": "call_terminal", "seq": seq, "ok": False, "call_id": call_id,
                               "error": f"{type(exc).__name__}: {exc}",
                               "latency_ms": round(1000 * (time.time() - t0), 3), "usage": usage})
            return ""
        text = res.text if isinstance(res, LLMResult) else res
        usage = dict(res.usage) if isinstance(res, LLMResult) else {}
        rsha = None
        if text:
            rsha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            (self._resp_dir / f"{rsha}.txt").write_bytes(text.encode("utf-8"))
        with self._lock:
            if not text:
                self.failures += 1
            self._account(usage)
            self._durable({"event": "call_terminal", "seq": seq, "ok": bool(text), "call_id": call_id,
                           "latency_ms": round(1000 * (time.time() - t0), 3),
                           "response_sha256": rsha, "usage": usage})
        return text or ""

    def close(self):
        self._fh.close()


def install_forced_mutation(select_operator, parent_selection, state, template):
    """L2 forced mutation bound to one sampler thread.

    Note: the forced parent is claimed atomically inside the
    operator choice and stashed thread-locally, so under samplers>1 the operator and
    the parent of the SAME sample are overridden together; another sampler
    interleaving between the two calls cannot steal or lose the binding."""
    lock = state.setdefault("lock", threading.Lock())
    tls = threading.local()
    state.setdefault("forced_overwritten", 0)

    def forced_select_operator():
        with lock:
            fp = state.get("forced_parent")
            state["forced_parent"] = None
        tls.forced = fp
        if fp is not None:
            return random.choice(["m1", "m2"]) if template == "mix" else template
        return select_operator()

    def forced_parent_selection(population, m):
        fp = getattr(tls, "forced", None)
        if fp is not None and m == 1:
            tls.forced = None
            with lock:
                state["forced_count"] += 1
            return [fp]
        return parent_selection(population, m)

    return forced_select_operator, forced_parent_selection


class OfflineClient:
    """$0 plumbing client routed THROUGH LLMShim: a fixed valid
    response as an LLMResult with zero usage. With variety=True the response cycles
    through distinct valid heuristics by call ordinal (X1 identity test)."""

    VARIETY = (
        ("Best fit: prefer the bin whose remaining capacity is closest to the item size.", "    return -(bins - item)\n"),
        ("Worst fit: prefer the bin with the largest remaining capacity.", "    return (bins - item).astype(float)\n"),
        ("First fit: prefer the lowest-index feasible bin.", "    return -np.arange(len(bins), dtype=float)\n"),
        ("Best fit with a tight-fit bonus below a gap of 5.", "    gap = bins - item\n    return -gap + np.where(gap < 5, 10.0, 0.0)\n"),
        ("Almost worst fit: largest remaining capacity, penalising the very largest.",
         "    gap = (bins - item).astype(float)\n    return gap - np.where(gap == gap.max(), 1000.0, 0.0)\n"),
    )

    def __init__(self, variety: bool = False):
        self.variety, self.n = variety, 0

    def __call__(self, prompt: str, **opts) -> LLMResult:
        desc, body = self.VARIETY[self.n % len(self.VARIETY) if self.variety else 0]
        self.n += 1
        return LLMResult(text=("{" + desc + "}\n```python\nimport numpy as np\n\n"
                               "def score(item: int, bins: np.ndarray) -> np.ndarray:\n" + body + "```\n"),
                         usage={"prompt_tokens": 0, "completion_tokens": 0, "charged_usd": 0.0,
                                "attempts": [{"ok": True, "charged_usd": 0.0}]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", required=True)
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--offline", action="store_true", help="offline client through the shim, real sandbox ($0)")
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--client-seed", type=int, default=None, help="default 40000+seed")
    ap.add_argument("--pop-size", type=int, default=PREREG_PARAMS["n"])
    ap.add_argument("--calls", type=int, default=400, help="hard LOGICAL LLM call cap (matched to the T-GADE arm)")
    ap.add_argument("--samples", type=int, default=None, help="EoH evolution-sample budget (default = --calls)")
    ap.add_argument("--no-think", dest="no_think", action="store_true", default=True)
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--cap-usd", type=float, default=0.5)
    ap.add_argument("--model", default=None, help="operator-model override (affinity study); requires --price-in/--price-out; provider fallbacks allowed")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider for the override model (no fallbacks); default: fallbacks allowed")
    ap.add_argument("--objective", default="train_c100", choices=["train_c100", "regime_max"],
                    help="EoH objective: canonical C=100 train set, or robust two-capacity max(mean excess C100 bank, C500 bank) (same evaluator as the T-GADE arm)")
    # survival rule swap, everything else EoH verbatim
    ap.add_argument("--survival", default="eoh", choices=["eoh", "thermo", "newest"],
                    help="'thermo' = removal-type thermodynamic rule (grant_evo.bench.eoh_thermo) in place of population_management")
    ap.add_argument("--exclusion", default="level", choices=["level", "level2", "genotype", "none"],
                    help="level = EoH objective dedupe; level2 = at most two per level; none = coexistence")
    ap.add_argument("--forced-template", default="m2", choices=["m2", "m1", "mix", "none"],
                    help="level2 only: the sample right after a second occupant is admitted is a forced mutation of it")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--carrier", default="hybrid", choices=["behaviour", "nl", "hybrid"])
    ap.add_argument("--hybrid-weights", default="0.8,0.1,0.1")
    ap.add_argument("--probe-len", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--eval-timeout", type=float, default=None,
                    help="lethal-candidate wall-clock cap (s) of one train evaluation and of the signature probe; default 120 (historical)")
    ap.add_argument("--signature-source", default="probe", choices=["probe", "energy"],
                    help="'energy' = behaviour signature as a by-product of the energy evaluation")
    ap.add_argument("--deterministic-only", action="store_true",
                    help="determinism gate: candidates using randomness APIs are invalid (objective None)")
    ap.add_argument("--operators", default=None,
                    help="baseline B1: comma list overriding EoH's operator set, e.g. 'i1' = parent-free "
                         "sampling only (best-of-N); default = EoH e1,e2,m1,m2 with equal weights")
    ap.add_argument("--parent-selection", default="rank", choices=["rank", "uniform"],
                    help="baseline B2: 'uniform' replaces EoH's rank-probability parent selection by uniform sampling")
    ap.add_argument("--samplers", type=int, default=1,
                    help="EoH sampler threads")
    ap.add_argument("--evaluators", type=int, default=1, help="EoH evaluator threads")
    ap.add_argument("--offline-variety", action="store_true",
                    help="offline client cycles through distinct valid heuristics (X1 identity test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume-from", default=None,
                    help="not supported in this release (refused); kept for command-line compatibility"
                         "REMAINING logical-call budget; the old ledger/responses are kept as *_part1")
    a = ap.parse_args(argv)
    P = PREREG_PARAMS
    if a.model and (a.price_in is None or a.price_out is None):
        raise SystemExit("REFUSED: --model needs --price-in and --price-out")
    OP_MODEL = a.model or P["model"]
    OP_PIN = a.price_in if a.model else P["price_in_usd_per_m"]
    OP_POUT = a.price_out if a.model else P["price_out_usd_per_m"]
    OP_PROV = ([a.provider] if a.provider else None) if a.model else P["provider_order"]
    OP_FB = (not a.provider) if a.model else P["allow_fallbacks"]
    if not a.offline and not a.confirm_paid:
        print("REFUSED: paid run needs --confirm-paid (or use --offline)")
        return 2
    samples = a.samples or a.calls

    from grant_evo.bench.adapters.bp_online import BpOnlineAdapter, EVAL_TIMEOUT, uses_randomness  # noqa: PLC0415  (stubs InterfaceLLM)
    from grant_evo.bench.sandbox import run_sandboxed, image_digest, PROTOCOL  # noqa: PLC0415
    from eoh.config import EoHConfig, LLMConfig  # noqa: PLC0415
    from eoh.eoh import eoh as eoh_mod  # noqa: PLC0415
    from eoh.eoh import evolution as evo_mod  # noqa: PLC0415
    from eoh.utils.createFolders import create_folders  # noqa: PLC0415
    from prob import BPONLINE  # noqa: PLC0415

    stamp = time.strftime("%Y%m%dT%H%M%S")
    resume = None
    if a.resume_from:
        print("REFUSED: --resume-from is not supported in this release; start a new run")
        return 2
    out = Path(a.out or (REPO / "runs" / (("offline_" if a.offline else "") + "eoh_" + a.label + "_" + stamp)))
    out.mkdir(parents=True, exist_ok=False)
    create_folders(str(out))

    if not check_seal(Path(a.bank)):
        print("REFUSED: gen0 bank does not match its seal")
        return 2
    bank = json.loads(Path(a.bank).read_bytes())
    seeds = [{"algorithm": g.get("thought", ""), "code": g["code"]} for g in bank["genotypes"]]
    seeds_path = out / "seeds.json"
    seeds_path.write_text(json.dumps(seeds, indent=1), encoding="utf-8", newline="\n")

    adapter = BpOnlineAdapter(train_k=5, items=5000, objective=a.objective, diversity_carrier=a.carrier,
                              hybrid_weights=tuple(float(x) for x in a.hybrid_weights.split(",")),
                              probe_len=a.probe_len, eval_timeout=a.eval_timeout,
                              signature_source=a.signature_source,
                              deterministic_only=a.deterministic_only)  # shared evaluator + endpoint banks (+ X1 rows)
    inst, lb = adapter._instances_payload()
    try:  # raw excess upper bound of the train evaluator (one item per bin); the survival energy clip
        # inst = {dataset: {instance: {...}}}, lb = {dataset: mean L1 bound}; one item per bin
        raw_upper_bound = float(np.mean([(np.mean([len(i["items"]) for i in insts.values()]) - float(lb[ds])) / float(lb[ds])
                                         for ds, insts in inst.items()]))
    except Exception:  # noqa: BLE001
        raw_upper_bound = None
    audit_lock = threading.Lock()
    audit_fh = open(out / "eval_audit.jsonl", "a", encoding="utf-8", newline="\n")
    eval_state = {"n": 0, "sig_meta": {}}  # code sha -> {eval_id, raw} of the evaluation that produced the cached signature

    def sandbox_raw(code: str) -> "float | None":
        if adapter.deterministic_only and uses_randomness(code):  # determinism gate, same rule as adapter.energy
            adapter.randomness_rejects += 1
            return None
        if a.objective == "regime_max":  # same robust objective as the T-GADE arm (raw, unclipped)
            comp = adapter._regime_raw(code)
            return None if comp is None else float(max(comp.values()))
        res = run_sandboxed({"op": "energy", "code": code, "capacity": 100, "instances": inst, "lb": lb,
                             **({"signature": True} if adapter.signature_source == "energy" else {})},
                            timeout=adapter.eval_timeout)
        if res is None or not res.get("ok"):
            return None
        v = res.get("value")
        v = None if v is None or not np.isfinite(v) or v < 0 else float(v)
        if adapter.signature_source == "energy":  # the same evaluation fills the row cache
            with audit_lock:  # signature and raw of ONE evaluation stored together under one eval_id
                eval_state["n"] += 1
                adapter._store_signature(code, res.get("signature"))
                eval_state["sig_meta"][hashlib.sha256(code.encode("utf-8")).hexdigest()] = {"eval_id": eval_state["n"], "raw": v}
        return v

    def sandbox_eval(problem, code, timeout):  # replaces the host-side spawn evaluator (injection 2)
        t_eval = time.time()
        raw = sandbox_raw(code)
        with audit_lock:
            audit_fh.write(json.dumps({"code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                                       "raw": raw, "ts": time.time(), "dt_s": round(time.time() - t_eval, 3),
                                       "thread": threading.current_thread().name}) + "\n")
            audit_fh.flush()
        return raw

    evo_mod._eval_with_timeout = sandbox_eval
    eoh_mod._eval_with_timeout = sandbox_eval

    problem = BPONLINE(capacity=100, timeout=EVAL_TIMEOUT)
    remaining_calls = a.calls - (resume["prior_calls"] if resume else 0)
    remaining_samples = samples - (resume["samples_done"] if resume else 0)
    if resume:
        cfg = EoHConfig(llm=LLMConfig(api_endpoint="injected", api_key="injected", model=OP_MODEL),
                        pop_size=a.pop_size, n_pop=max(1, math.ceil(samples / a.pop_size)),
                        num_samplers=a.samplers, num_evaluators=a.evaluators, max_sample_nums=max(1, remaining_samples),
                        output_dir=str(out), use_seed=False, use_continue=True,
                        continue_path=resume["continue_path"], continue_id=resume["continue_id"])
    else:
        cfg = EoHConfig(llm=LLMConfig(api_endpoint="injected", api_key="injected",
                                      model=OP_MODEL),
                        pop_size=a.pop_size, n_pop=max(1, math.ceil(samples / a.pop_size)),
                        num_samplers=a.samplers, num_evaluators=a.evaluators, max_sample_nums=samples,
                        output_dir=str(out), use_seed=True, seed_path=str(seeds_path))
    if a.operators:  # B1: operator set override (weights equal); EOH reads config.operators at construction
        cfg.operators = [x.strip() for x in a.operators.split(",") if x.strip()]
        cfg.operator_weights = [1.0] * len(cfg.operators)
    eoh = eoh_mod.EOH(cfg, problem)

    _orig_evaluate_seeds = eoh.evolution.evaluate_seeds

    def evaluate_seeds_sorted(seed_list):  # wrapper 3
        pop = sorted(_orig_evaluate_seeds(seed_list), key=lambda x: x["objective"])
        if pop:
            eoh._best_obj = pop[0]["objective"]
        return pop

    eoh.evolution.evaluate_seeds = evaluate_seeds_sorted

    if a.offline:
        cseed = None
        client = OfflineClient(variety=a.offline_variety)
    else:
        from grant_evo.bench.clients import OpenRouterClient  # noqa: PLC0415
        P = PREREG_PARAMS
        cseed = a.client_seed if a.client_seed is not None else 40000 + a.seed
        client = OpenRouterClient(OP_MODEL, temperature=P["t_sample"], budget_usd=a.cap_usd,
                                  retries=P["transport_retries"], min_interval_s=P["min_interval_s"],
                                  max_tokens=P["max_tokens"], price_in_usd_per_m=OP_PIN,
                                  price_out_usd_per_m=OP_POUT,
                                  max_input_bytes=P["max_input_bytes"],
                                  chat_overhead_tokens=P["chat_overhead_tokens"],
                                  provider_order=OP_PROV, allow_fallbacks=OP_FB,
                                  seed=cseed)
    reg_state = {"seq": 0, "fh": open(out / "registration.jsonl", "w", encoding="utf-8", newline="\n")}
    shim = LLMShim(client, max(0, remaining_calls), out, a.no_think, version=lambda: reg_state["seq"])
    eoh.evolution.llm = shim  # injection point 1

    # ── survival-rule swap ──────────
    survival_state = {"fatal": None, "rows_cache": {}, "trace_fh": None}
    _orig_pm = eoh_mod.population_management
    if a.survival == "newest":  # B2: no survival selection, keep the N most recently registered valid members
        def newest_management(pop, size):
            valid = [x for x in pop if x.get("objective") is not None]
            return valid[-size:]
        eoh_mod.population_management = newest_management
    if a.parent_selection == "uniform":  # B2: no parent selection pressure either (restored in the finally block)
        survival_state["orig_ps"] = evo_mod.parent_selection
        evo_mod.parent_selection = lambda population, m: random.sample(population, min(m, len(population)))
    if a.survival == "thermo":
        from grant_evo.bench.eoh_thermo import thermo_management  # noqa: PLC0415
        from grant_evo.bench.adapters.bp_online import RAW_MAX  # noqa: PLC0415
        feat = f"{a.carrier}|{a.hybrid_weights}|probe{a.probe_len}|{adapter.energy_spec}"
        survival_state["trace_fh"] = open(out / "survival_trace.jsonl", "w", encoding="utf-8", newline="\n")
        row_lock = threading.Lock()

        def rows(ind):  # cache key = code + description + feature settings (never code alone)
            key = hashlib.sha256((ind["code"] + "\x00" + (ind.get("algorithm") or "") + "\x00" + feat)
                                 .encode("utf-8")).hexdigest()
            with row_lock:
                if key in survival_state["rows_cache"]:
                    return survival_state["rows_cache"][key]
            m = None
            for _attempt in range(2):  # measurement failure: one re-measure, then fail-fast
                m = adapter.diversity_matrix({"code": ind["code"], "thought": ind.get("algorithm") or ""})
                if m is not None:
                    break
            if m is None:
                survival_state["fatal"] = ("neighbourhood measurement failed twice for code "
                                           + hashlib.sha256(ind["code"].encode("utf-8")).hexdigest()[:16])
                shim.max_calls = 0  # stop the run: every further LLM call is refused
            with row_lock:
                survival_state["rows_cache"][key] = m
            meta = eval_state["sig_meta"].get(hashlib.sha256(ind["code"].encode("utf-8")).hexdigest())
            if meta is not None and ind.get("objective") is not None and (
                    meta["raw"] is None or abs(round(meta["raw"], 5) - float(ind["objective"])) > 1e-5):
                survival_state["sig_mismatch"] = survival_state.get("sig_mismatch", 0) + 1  # re-evaluated code gave another objective
            return m

        def energy(ind):
            return min(float(ind["objective"]), RAW_MAX) / RAW_MAX

        def survival_hook(pop, size):
            if survival_state["fatal"] is not None:
                return _orig_pm(pop, size)
            trace = []
            try:
                res = thermo_management(pop, size, temperature=a.temperature, rows=rows, energy=energy,
                                        exclusion=a.exclusion, eps=a.eps, trace=trace)
            except Exception as exc:  # safety valve: the upstream worker would swallow this exception
                survival_state["fatal"] = f"survival rule failed: {type(exc).__name__}: {exc}"
                shim.max_calls = 0  # stop the run: every further LLM call is refused
                newest = pop[-1] if pop else None
                return _orig_pm([x for x in pop if x is not newest], size)  # keep the pre-violation population
            if survival_state["fatal"] is not None:  # a row failed inside this call: not a valid decision
                return _orig_pm(pop, size)
            for t in trace:
                t["sample"] = eoh._sample_count
                survival_state["trace_fh"].write(json.dumps(t) + "\n")
            survival_state["trace_fh"].flush()
            if a.exclusion == "level2" and a.forced_template != "none" and trace and trace[-1]["second_occupant_admitted"]:
                with survival_state.setdefault("lock", threading.Lock()):
                    if survival_state.get("forced_parent") is not None:
                        survival_state["forced_overwritten"] = survival_state.get("forced_overwritten", 0) + 1
                    survival_state["forced_parent"] = pop[-1]  # the newcomer: next sample = forced mutation of it
            return res

        eoh_mod.population_management = survival_hook  # module-global lookup at call time (init + steady state)
        if a.exclusion == "level2" and a.forced_template != "none":

            # mutation (operator and parent of the NEXT sample are overridden once; call budget unchanged)
            survival_state["forced_parent"], survival_state["forced_count"] = None, 0
            survival_state["orig_ps"] = evo_mod.parent_selection
            eoh._select_operator, evo_mod.parent_selection = install_forced_mutation(
                eoh._select_operator, evo_mod.parent_selection, survival_state, a.forced_template)

    inner_pm = eoh_mod.population_management  # verbatim EoH rule or the X1 survival hook

    def registration_log(pop, size):
        """Every registration (init + steady state) in order, with the sampler's call seq and the population
        version it sampled from ( sample ID / snapshot version / registration
        order must be recorded before any sampler>1 cohort). pop_in_full is kept for the first call only (replay)."""
        res = inner_pm(pop, size)
        reg_state["seq"] += 1
        cs, pv = getattr(shim.tls, "seq", None), getattr(shim.tls, "pop_version", None)
        rec = {"reg_seq": reg_state["seq"], "call_seq": cs, "pop_version_at_call": pv,
               "snapshot_age": None if pv is None else reg_state["seq"] - 1 - pv,
               "thread": threading.current_thread().name, "n_in": len(pop), "ts": time.time(),
               "newcomer": {k: pop[-1].get(k) for k in ("code", "algorithm", "objective")} if pop else None,
               "pop_after": [[x.get("objective"), hashlib.sha256(x["code"].encode("utf-8")).hexdigest()] for x in res]}
        if reg_state["seq"] == 1:
            rec["pop_in_full"] = [{k: x.get(k) for k in ("code", "algorithm", "objective")} for x in pop]
        reg_state["fh"].write(json.dumps(rec) + "\n")
        reg_state["fh"].flush()
        return res

    eoh_mod.population_management = registration_log

    banks = {"c100": _load_bank("confirmation_bank.json"), "c500": _load_bank("confirmation_bank_c500.json")}
    penalties = {k: _bank_penalty(b) for k, b in banks.items()}
    # the exclusion argument is applied only by the thermodynamic survival rule; record the rule actually used
    exclusion_eff = {"eoh": "level", "newest": "none", "thermo": a.exclusion}[a.survival]
    design = {"arm": "EOH_official_commit_pinned", "label": a.label, "offline": a.offline, "seed": a.seed,
              "client_seed": cseed, "eoh_search_seed": "upstream random.seed(2024), fixed",
              "bank": bank["sha256"], "pop_size": a.pop_size, "logical_call_cap": a.calls,
              "sample_budget": samples, "no_think": a.no_think, "operators": cfg.operators,
              "n_parents": cfg.n_parents, "update": {"eoh": "steady-state (upstream), objective-dedup, 5-decimal rounding",
                                                     "thermo": "steady-state, removal-type thermodynamic survival rule, 5-decimal rounding",
                                                     "newest": "steady-state, keep the N most recently registered valid members, 5-decimal rounding"}[a.survival],
              "injection": ["Evolution.llm", "evolution._eval_with_timeout", "eoh._eval_with_timeout"],
              "objective": a.objective, "energy_spec": adapter.energy_spec,
              "exclusion_requested": a.exclusion, "exclusion_effective": exclusion_eff,
              "exclusion_rule": {"level": "one individual per objective-function value (EoH rule)", "level2": "at most two per objective-function value",
                                 "genotype": "one individual per genotype (algorithm, code) exact match", "none": "no exclusion"}[exclusion_eff],
              "genotype_key": "(algorithm, code) exact match" if exclusion_eff == "genotype" else "not applicable",
              "wrappers": ["evaluate_seeds output sorted by objective (multiset unchanged); best tracker seeded",
                           "final selection = min finite objective of the last saved population",
                           "/no_think prompt prefix" if a.no_think else "no prompt prefix"],
              "runtime_side_effects": ["BpOnlineAdapter._lazy_imports stubs evolution.InterfaceLLM "
                                       "(constructor ping suppressed), replaced by the shim"],
              "upstream_commit": _git_head(REPO / "third_party/EoH"), "code_commit": _git_head(REPO),
              "model": None if a.offline else OP_MODEL, "operator_model_override": bool(a.model),
              "execution": None if a.offline else {k: PREREG_PARAMS[k] for k in
                                                   ("provider_order", "allow_fallbacks", "min_interval_s",
                                                    "transport_retries", "max_tokens", "t_sample")},
              "banks": {k: v.get("_sha256") for k, v in banks.items()}, "penalties_raw": penalties,
              "sandbox_image": image_digest(), "sandbox_protocol": PROTOCOL, "cap_usd_per_run": a.cap_usd,
              "note": "commit-pinned EoH implementation connected to the common environment; "
                      "not a reproduction of the 2024 paper run",
              "survival": a.survival, "exclusion": a.exclusion, "temperature": a.temperature,
              "carrier": a.carrier, "hybrid_weights": a.hybrid_weights, "probe_len": a.probe_len, "eps": a.eps,
              "free_energy_mode": "extensive",
              "survival_energy": "E = min(objective_rounded5, 2.0) / 2.0 (T-GADE clip); parent selection = EoH rank of the same objective",
              "raw_upper_bound_train": raw_upper_bound,
              "eval_timeout_s": adapter.eval_timeout, "signature_source": a.signature_source,
              "deterministic_only": a.deterministic_only,
              "num_samplers": a.samplers, "num_evaluators": a.evaluators,
              "parent_selection": a.parent_selection, "operators_override": a.operators,
              "forced_template": (a.forced_template if a.exclusion == "level2" else None),
              "selection_rule": "history best over valid evaluated samples + seeds (v2 #2); final-population best recorded as secondary",
              "neighbourhood_failure_rule": "one re-measure, then the run is stopped and marked aborted (v2 #4)",
              "resume": resume}
    dpath = out / ("design_resume_" + stamp + ".json" if resume else "design.json")
    write_seal(dpath, _atomic_json(dpath, design))

    t0 = time.time()
    try:
        eoh.run()
    finally:
        shim.close()
        reg_state["fh"].close()
        eoh_mod.population_management = _orig_pm  # restore the module globals (tests call main() repeatedly)
        if "orig_ps" in survival_state:
            evo_mod.parent_selection = survival_state["orig_ps"]
    elapsed = time.time() - t0

    pops = sorted((out / "results" / "pops").glob("population_generation_*.json"),
                  key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    final = json.loads(pops[-1].read_text(encoding="utf-8")) if pops else []
    finite = [x for x in final if x.get("objective") is not None and np.isfinite(x["objective"])]
    final_best = min(finite, key=lambda x: x["objective"]) if finite else None
    # history best (v2 #2): every valid evaluated sample + the gen-0 seeds; first arrival wins ties
    hist = []
    for sp in sorted((out / "results" / "samples").glob("samples_*~*.json"),
                     key=lambda p: int(p.stem.split("_", 1)[1].split("~")[0])):
        hist.extend(x for x in json.loads(sp.read_text(encoding="utf-8"))
                    if x.get("objective") is not None and np.isfinite(x["objective"]) and x.get("code"))
    if pops:
        hist = [x for x in json.loads(pops[0].read_text(encoding="utf-8"))
                if x.get("objective") is not None and np.isfinite(x["objective"])] + hist
    hist_best = min(hist, key=lambda x: x["objective"]) if hist else None
    best = hist_best if hist_best is not None else final_best
    genes = None if best is None else {"code": best["code"], "thought": best.get("algorithm") or ""}
    endpoint = _endpoint(adapter, genes, banks, penalties)
    train_raw = None if genes is None else sandbox_raw(genes["code"])  # unclipped raw, common evaluator
    summary = {"arm": design["arm"], "label": a.label, "elapsed_s": round(elapsed, 1),
               "logical_calls": shim.calls, "physical_attempts": shim.physical_attempts,
               "calls_refused_after_cap": shim.refused, "call_failures": shim.failures,
               "samples_recorded": eoh._sample_count, "usd": round(shim.usd, 6),
               "prompt_tokens": shim.prompt_tokens, "completion_tokens": shim.completion_tokens,
               "best_objective_rounded": None if best is None else best["objective"],
               "history_best_objective": None if hist_best is None else hist_best["objective"],
               "final_best_objective": None if final_best is None else final_best["objective"],
               "selected_from": "history" if hist_best is not None else "final",
               "survival": a.survival, "exclusion": a.exclusion, "temperature": a.temperature,
               "aborted": survival_state["fatal"], "forced_mutations": survival_state.get("forced_count"),
               "forced_overwritten": survival_state.get("forced_overwritten"),
               "num_samplers": a.samplers, "num_evaluators": a.evaluators,
               "registrations": reg_state["seq"], "signature_objective_mismatch": survival_state.get("sig_mismatch", 0),
               "evaluations_with_signature": eval_state["n"],
               "randomness_rejects": adapter.randomness_rejects,
               "train_raw_excess_best_unclipped": train_raw,
               "final_population_size": len(final),
               "final_population_objectives": [x.get("objective") for x in final],
               "endpoint": endpoint, "generations_saved": len(pops)}
    write_seal(out / "summary.json", _atomic_json(out / "summary.json", summary))
    if genes is not None:
        write_seal(out / "selected.json", _atomic_json(out / "selected.json",
                   {"genes": genes, "endpoint": endpoint}))
    print(json.dumps({k: summary[k] for k in ("logical_calls", "physical_attempts", "calls_refused_after_cap",
                                              "samples_recorded", "best_objective_rounded",
                                              "train_raw_excess_best_unclipped", "final_population_size",
                                              "usd", "elapsed_s")} | {"c100": endpoint["c100"]["raw"],
                                                                      "c500": endpoint["c500"]["raw"]}))
    audit_fh.close()
    if survival_state["trace_fh"] is not None:
        survival_state["trace_fh"].close()
    if survival_state["fatal"] is not None:
        print("ABORTED", survival_state["fatal"], out)
        return 3
    print("DONE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
