"""T-GADE generational loop.

THE evolution loop. Lives in the kernel package so problem
adapters/orchestration (grant_evo.bench) cannot fork it. Selection is
delegated to grant_evo.tgade.selection.thermodynamical_select.

Topology per generation t -> t+1 (|P'| <= 2N+1):
  Step 3  elite carryover: fresh id, source_id = elite, genes copied, exempt
          from Steps 4-6 (re-enters evaluation/selection unchanged).
  Step 4  crossover: disjoint pairing of the N parents into N/2 pairs, one
          child per (primary, secondary) order => N children; each child
          inherits the primary parent's host model.
  Step 5  mutation over the UNION of parent variants (N copies) and children
          (N): adapter.mutate() for every union member; identity output is
          legal (trade-off inapplicable), None is failure => vacancy.
  Step 6  integrity: when the adapter defines integrity(), it runs for
          every union member with that individual own-host client; None is
          failure => vacancy.
  Operator failure => vacancy + Event (never a silent parent copy).
  Evaluation: validate -> energy -> diversity; missing => excluded + Event
  (missing is never zero, C7). < N valid candidates => quarantine evidence is
  persisted atomically, then PopulationExtinctionError.
  Selection: boson thermodynamical_select; survivors materialised with an
  explicit instance mapping table (C3).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from typing import Callable

from grant_evo.tgade.selection import thermodynamical_select

# ── leaf types (owned here so the kernel package has no bench dependency;
# circular import engine->bench->core->engine) ─────
LLMClient = Callable[[str], "str | None"]


@dataclass
class LLMResult:
    """Typed client result. Clients may return this instead
    of a bare string; the engine normalises both. usage keys (all optional):
    prompt_tokens, completion_tokens, cost_usd, provider, retries."""

    text: "str | None"
    usage: dict = field(default_factory=dict)


class UnknownHostError(KeyError):
    """Requested host has no registered client. Fail closed:
    a silent fallback would make the recorded route a lie."""


@dataclass
class CallRecord:
    """In-memory merged view of one LLM call (durable form = two JSONL events,
    call_started flushed BEFORE transport, call_terminal after;)."""

    seq: int
    op_id: str
    gen: int
    role: str  # "init" | "mutate" | "cross" | "integrity" | "other"
    individual_id: "str | None"
    requested_host: "str | None"
    resolved_host: str
    prompt_sha256: str  # full 64-hex
    input_digests: list = field(default_factory=list)
    response_sha256: "str | None" = None
    ok: bool = False
    latency_ms: "float | None" = None
    error: "str | None" = None
    usage: dict = field(default_factory=dict)  # declared fields only
    opts: dict = field(default_factory=dict)  # per-call sampling options (e.g. temperature)


# usage fields the engine recognises; everything else in LLMResult.usage is
# kept verbatim on the row but NEVER aggregated.
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "cost_usd", "retries")
USAGE_META_FIELDS = ("provider", "model", "request_id", "api_seed", "latency_ms",
                     "finish_reason", "attempts", "error")  # attempts: every
#   physical retry of one logical call is persisted in the ledger row


def viability(adapter, genes: dict):
    """ONE viability check shared by the engine, the gen0-bank producer and
    MULTISTART: validate -> energy in [0,1] ->
    diversity matrix of exact (M, D) shape. Returns
    (ok, kind, reason, energy, matrix); kind in {"invalid",
    "missing_energy", "missing_diversity"} when not ok."""
    valid, why = adapter.validate(genes)
    if not valid:
        return False, "invalid", why, None, None
    e = adapter.energy(genes)
    if e is None:
        return False, "missing_energy", "energy=None", None, None
    if not (0.0 <= float(e) <= 1.0):
        return False, "invalid", f"energy {e!r} outside [0,1] (adapter must normalise, C5)", None, None
    m = adapter.diversity_matrix(genes)
    if m is None:
        return False, "missing_diversity", "diversity=None", None, None
    m = np.asarray(m, dtype=float)
    if m.shape != (adapter.num_sections, adapter.dim):
        return False, "invalid", f"diversity shape {m.shape} != (M, D)", None, None
    return True, None, "ok", float(e), m


def loci_canonical_digest(loci: dict) -> str:
    """THE genotype digest: sha256 over the
    canonical serialization of the EXPORTED loci view — sorted key=value
    lines, LF-terminated, UTF-8. Rust recomputes the identical bytes from the
    trace genome, so operator chain, candidate genome, survivors and parents
    all live in ONE cross-language digest domain."""
    canon = "".join(f"{k}={loci[k]}" + chr(10) for k in sorted(loci))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def genotype_key(loci: dict) -> str:
    """Occupancy key for Fermi-type selection: sha256 over the canonical JSON of the
    exported loci view. JSON escaping keeps the field boundaries unambiguous, so two
    different (description, code) pairs never share a key (the key=value line form of
    ``loci_canonical_digest`` can conflate values that contain a newline followed by
    another field name). ``loci_canonical_digest`` stays the bank / seal / trace digest
    and is not used for occupancy."""
    blob = json.dumps(loci, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      default=_np_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Individual:
    """One member of a population or candidate pool. Genes are adapter-opaque."""

    id: str
    genes: dict
    gen: int
    op: str  # "init" | "elite_carryover" | "parent_variant" | "cross" | "survivor" | "clone"
    parent_ids: list[str] = field(default_factory=list)
    source_id: str | None = None  # originating individual for clones/elite
    host: str | None = None  # bound host-model name
    energy: float | None = None  # E in [0,1]; None = missing (C7)
    meta: dict = field(default_factory=dict)


@dataclass
class Event:
    """Operator/evaluation evidence record."""

    gen: int
    kind: str  # "operator_failure" | "invalid" | "missing_energy" | "missing_diversity"
    individual_id: str | None
    op: str | None
    parent_ids: list[str]
    reason: str


ProblemAdapter = "grant_evo.bench.adapter.ProblemAdapter"  # structural typing only


class PopulationExtinctionError(RuntimeError):
    """Raised when fewer than N valid candidates remain after fail-closed
    exclusion. The run is QUARANTINED: mass
    operator/evaluation failure signals a broken instrument, and continuing
    by boson duplication would hide it."""


@dataclass
class BenchConfig:
    n: int = 8
    generations: int = 5
    temperature: float = 0.5
    strength: str = "mid"  # operator strength dial (M2 ladder)
    eps: float = 1e-3
    free_energy_mode: str = "extensive"
    occupancy: str = "boson"
    seed: int = 0
    hosts: tuple = (None,)  # host-model names cycled over gen0 (Step 1)
    parent_policy: str = "mutate_all"  # Step 5 treatment of the N survivors:
    #   "mutate_all" = Algorithm 1 canonical (every survivor enters the pool
    #   only as its mutated variant; the elite is carried unmutated);
    #   "keep_originals_mutate_duplicates": the
    #   FIRST occurrence of each survivor genotype enters the pool unchanged
    #   (no LLM call), later occurrences (bosonic duplicates) are mutated
    #   (+ integrity) instead; no elite carry-over (originals are kept), so
    #   the pool is 2N. Explicit variant: manifest-recorded, _UNCERTIFIED.
    child_post_ops: str = "mutate+integrity"  # post-crossover stack applied to
    #   the crossover children: "mutate+integrity" =
    #   Algorithm 1 canonical (Step 5 mutation, then Step 6 integrity);
    #   "integrity" = mutation-only isolation (children skip Step 5, keep Step 6);
    #   "none" = post-cross stack removed (children enter the pool as the
    #   crossover output; EoH one-sample-one-operator convention). Parent
    #   variants always get mutate+integrity. Non-canonical values are EXPLICIT
    #   ablations: manifest-recorded via config_sha256 and _UNCERTIFIED, like
    #   integrity_mode="skip".
    parent_alloc: str = "uniform"  # "rank" = EoH parent
    #   allocation (∝ 1/(rank+1+N) over the E-sorted population, WITH replacement) for the N
    #   mutation targets and the N crossover pairs; pool composition (elite 1 + variants N +
    #   children N) unchanged. "uniform" = Algorithm 1 canonical (each survivor mutated once,
    #   disjoint pairing). Requires llm_workers >= 1 (batched path only).
    operator_policy: str = "tgade"  # "eoh" = the N survivors enter the pool
    #   UNCHANGED (no calls, no elite carry-over) and N+1 offspring are generated EoH-style:
    #   operator drawn uniformly from e1/e2/m1/m2, parents by rank allocation, no post-ops
    #   (Step 6 integrity only when integrity_mode == "full"). Pool = 2N+1. Whole-configuration
    #   variant: manifest-recorded, _UNCERTIFIED. Requires llm_workers >= 1.
    gen0_integrity: bool = False  # Step-6 repair also on gen0 (opt-in: adds N host calls at gen 0)
    llm_workers: int = 0  # 0 = legacy sequential operator calls; >=1 = batched
    # parallel path with pre-assigned op ids and per-op rng streams (rng_schedule_version per_op_v2)
    rng_schedule_version: str = "shared_v1"
    integrity_mode: str = "full"  # "full": Step 6 REQUIRED (adapter.integrity must
    #   exist and run per union member on its own host); "skip": EXPLICIT ablation,
    #   recorded in the manifest.
    out_dir: str | None = None  # quarantine/evidence directory (P0-5)
    oracle_path: str | None = None  # tgade-contract-oracle exe; default = repo build
    gen0_bank_sha256: str | None = None  # shared gen0 bank identity (E4)
    gen0_bank_digests: tuple = ()  # canonical digests of the bank genotypes
    parse_predicate: str | None = None  # adapter parse-failure predicate name,
    #   bound into config_sha256 so the oracle can replay parse vacancies

    def __post_init__(self) -> None:
        if self.n < 2 or self.n % 2 != 0:
            raise ValueError("n must be an even integer >= 2 (disjoint pairing)")
        if self.integrity_mode not in ("full", "skip"):
            raise ValueError(f"unknown integrity_mode: {self.integrity_mode!r}")
        if self.child_post_ops not in ("mutate+integrity", "integrity", "none"):
            raise ValueError(f"unknown child_post_ops: {self.child_post_ops!r}")
        if self.parent_policy not in ("mutate_all", "keep_originals_mutate_duplicates"):
            raise ValueError(f"unknown parent_policy: {self.parent_policy!r}")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.parent_alloc not in ("uniform", "rank"):
            raise ValueError(f"unknown parent_alloc: {self.parent_alloc!r}")
        if self.operator_policy not in ("tgade", "eoh"):
            raise ValueError(f"unknown operator_policy: {self.operator_policy!r}")
        if (self.parent_alloc == "rank" or self.operator_policy == "eoh") and self.llm_workers < 1:
            raise ValueError("parent_alloc='rank' / operator_policy='eoh' need the batched path (llm_workers >= 1)")
        if self.operator_policy == "eoh" and self.parent_policy != "mutate_all":
            raise ValueError("operator_policy='eoh' keeps the survivors itself; parent_policy must stay 'mutate_all'")


def _usage_totals(ledger) -> dict:
    tot: dict = {}
    for c in ledger:
        for k in USAGE_FIELDS:
            v = c.usage.get(k)
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    return tot


@dataclass
class RunResult:
    population: list[Individual]
    lineage: list[Individual]
    events: list[Event]
    generation_log: list[dict]
    manifest: dict
    call_ledger: "list[CallRecord]" = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "population": [asdict(i) for i in self.population],
                "lineage": [asdict(i) for i in self.lineage],
                "events": [asdict(e) for e in self.events],
                "generation_log": self.generation_log,
                "manifest": self.manifest,
                "call_ledger": [asdict(c) for c in self.call_ledger],
            },
            indent=1,
            ensure_ascii=False,
            default=_np_default,
        )


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


import functools


@functools.lru_cache(maxsize=16)
def _git_state(repo: Path) -> dict:
    """Best-effort commit + dirty digest for the manifest (C8). Cached per repo."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30,
        ).stdout
        return {
            "git_commit": head or None,
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        }
    except Exception:  # noqa: BLE001 - manifest stays best-effort offline
        return {"git_commit": None, "dirty_diff_sha256": None}


class BenchRun:
    """Owns ids, lineage, events, and the generation loop for one run."""

    def __init__(self, adapter, cfg: BenchConfig, llm, gen0_bank=None,
                 eval_workers: int = 1):
        # eval_workers (V101 P2): concurrency of sandbox viability checks.
        # Execution-only knob: values, event order and ledger bytes are
        # unchanged, so it is deliberately NOT part of BenchConfig/config_sha256.
        self._eval_workers = max(1, int(eval_workers))
        # llm: one LLMClient or a registry {host_name: LLMClient}, optional
        # "default" key.
        # gen0_bank: {sha256, digests, genotypes} shared across arms (E4);
        # cfg.gen0_bank_sha256/digests must match it (fail-closed).
        self.adapter = adapter
        self.cfg = cfg
        self._gen0_bank = gen0_bank
        if cfg.parse_predicate is None:
            cfg.parse_predicate = getattr(adapter, "parse_predicate", None)
        if cfg.integrity_mode == "full" and not cfg.parse_predicate:
            raise ValueError(
                f"adapter {adapter.name!r} declares no parse_predicate (required "
                "for vacancy-aware certification)")
        if (gen0_bank is None) != (cfg.gen0_bank_sha256 is None):
            raise ValueError("gen0_bank and cfg.gen0_bank_sha256 must be "
                             "supplied together")
        if gen0_bank is not None:
            from grant_evo.bench.multistart import bank_digest  # noqa: PLC0415
            actual_digests = [self._geno_digest(g)
                              for g in gen0_bank["genotypes"]]
            if (bank_digest(gen0_bank["digests"]) != cfg.gen0_bank_sha256
                    or gen0_bank["sha256"] != cfg.gen0_bank_sha256
                    or sorted(cfg.gen0_bank_digests) != sorted(gen0_bank["digests"])
                    or len(gen0_bank["genotypes"]) != cfg.n
                    or len(gen0_bank["digests"]) != cfg.n
                    or sorted(actual_digests) != sorted(gen0_bank["digests"])):
                raise ValueError("gen0 bank identity mismatch (fail-closed)")
        # Registry roles:
        #   "operator"      -> Steps 4-5 (crossover + trade-off mutation)
        #   host name       -> Step 1 init and Step 6 integrity
        #   "default"       -> fallback for BOTH roles when the specific key is
        #                      absent (single-client setups stay valid).
        self._registry = llm if isinstance(llm, dict) else {"default": llm}
        # Code identity is captured at START: a commit
        # made while a run is in flight must not be reported as the run's code.
        self._git_at_start = _git_state(Path(__file__).resolve().parents[3])
        if cfg.integrity_mode == "full" and not callable(getattr(adapter, "loci_view", None)):
            raise ValueError(
                f"integrity_mode='full' requires adapter {adapter.name!r} to define "
                "loci_view(genes) (canonical genotype digest domain)")
        if cfg.integrity_mode == "full" and not callable(getattr(adapter, "integrity", None)):
            raise ValueError(
                f"integrity_mode='full' but adapter {adapter.name!r} defines no "
                "integrity(); pass integrity_mode='skip' to run the EXPLICIT "
                "no-Step-6 ablation")
        self.call_ledger: list[CallRecord] = []
        self._call_seq = 0
        self._op_output: dict[str, str] = {}  # op_id -> output genotype digest
        # per-thread operator context + one re-entrant I/O lock for ledger/seq/op ids
        self._tls = threading.local()
        self._ctx_default = {"gen": 0, "role": "other", "individual_id": None,
                             "op_id": "op0", "input_digests": []}
        self._io_lock = threading.RLock()
        if cfg.llm_workers >= 1 and cfg.rng_schedule_version != "per_op_v2":
            raise ValueError("llm_workers>=1 requires rng_schedule_version='per_op_v2' (declared, not implied)")
        self._ledger_path = None
        self._ledger_fh = None
        if cfg.out_dir:
            out = Path(cfg.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._ledger_path = out / (
                f"call_ledger_{adapter.name}_seed{cfg.seed}.jsonl")
            if self._ledger_path.exists():
                raise FileExistsError(
                    f"no-clobber: {self._ledger_path} already exists (S1-4)")
            self._ledger_fh = open(self._ledger_path, "a", encoding="utf-8",
                                   newline="\n")  # no CRLF translation: sha == disk
        self.rng = random.Random(cfg.seed)
        self._next = 0
        self.lineage: list[Individual] = []
        self.events: list[Event] = []
        self.generation_log: list[dict] = []

    @property
    def _ctx(self) -> dict:
        return getattr(self._tls, "ctx", self._ctx_default)

    @_ctx.setter
    def _ctx(self, value: dict) -> None:
        self._tls.ctx = value

    def _persist_response(self, sha: str, text: str) -> None:
        """Content-addressed raw response store next to the ledger
        (responses/<sha256>.txt), fsynced BEFORE the terminal row."""
        if self._ledger_path is None:
            return
        rdir = self._ledger_path.parent / "responses"
        rdir.mkdir(exist_ok=True)
        path = rdir / f"{sha}.txt"
        if path.exists():
            return
        tmp = path.with_name(f"{sha}.{threading.get_ident()}.tmp")  # unique per thread
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            import os as _os
            _os.fsync(fh.fileno())
        tmp.replace(path)

    def _durable(self, obj: dict) -> None:
        if self._ledger_fh is not None:
            with self._io_lock:
                self._ledger_fh.write(json.dumps(obj, ensure_ascii=False,
                                                 default=_np_default) + "\n")
                self._ledger_fh.flush()
                import os as _os
                _os.fsync(self._ledger_fh.fileno())

    def _client(self, host) -> LLMClient:
        if host in ("operator", None):
            key = "operator" if host == "operator" else None
            if key is not None and key in self._registry:
                raw, resolved = self._registry[key], key
            elif "default" in self._registry:
                raw, resolved = self._registry["default"], "default"
            else:
                raise UnknownHostError(
                    f"role {host!r} has no client and no default is registered")
        elif host in self._registry:
            raw, resolved = self._registry[host], host
        else:
            raise UnknownHostError(
                f"host {host!r} has no registered client (fail-closed; "
                f"known: {sorted(self._registry)})")

        def recording(prompt: str, **opts):
            # opts: per-call sampling options declared by the adapter (the
            # M2 strength ladder is a TEMPERATURE dial); recorded durably.
            if self.cfg.llm_workers >= 1 and getattr(raw, "accepts_call_id", False):
                # schedule-independent provider seed. An adapter-supplied call_id (e.g. the
                # content-derived repair key) takes precedence over the op_id default, so a
                # single-flight winner does not change the seed.
                opts = {"call_id": self._ctx["op_id"], **opts}
            elif "call_id" in opts and not getattr(raw, "accepts_call_id", False):
                opts = {k: v for k, v in opts.items() if k != "call_id"}
            with self._io_lock:
                self._call_seq += 1
                seq_now = self._call_seq
            row = CallRecord(
                seq=seq_now,
                op_id=self._ctx["op_id"],
                gen=self._ctx["gen"],
                role=self._ctx["role"],
                individual_id=self._ctx["individual_id"],
                requested_host=host,
                resolved_host=resolved,
                prompt_sha256=hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
                input_digests=list(self._ctx["input_digests"]),
                opts=dict(opts),
            )
            with self._io_lock:
                self.call_ledger.append(row)
            self._durable({"event": "call_started", "ts": time.time(), **{
                k: getattr(row, k) for k in (
                    "seq", "op_id", "gen", "role", "individual_id",
                    "requested_host", "resolved_host", "prompt_sha256",
                    "input_digests", "opts")}})
            t0 = time.time()
            try:
                out = raw(prompt, **opts) if opts else raw(prompt)
            except Exception as exc:
                row.latency_ms = round(1000 * (time.time() - t0), 3)
                row.error = f"{type(exc).__name__}: {exc}"
                # cost already incurred by earlier physical attempts of this
                # logical call (BudgetExceeded carries it;)
                row.usage = dict(getattr(exc, "usage", None) or {})
                self._durable({"event": "call_terminal", "seq": row.seq,
                               "ok": False, "error": row.error,
                               "latency_ms": row.latency_ms, "usage": row.usage})
                raise
            row.latency_ms = round(1000 * (time.time() - t0), 3)
            if isinstance(out, LLMResult):
                text = out.text
                row.usage = {k: out.usage.get(k) for k in
                             (*USAGE_FIELDS, *USAGE_META_FIELDS)
                             if out.usage.get(k) is not None}
            else:
                text = out
            if text is not None:
                row.ok = True
                row.response_sha256 = hashlib.sha256(
                    str(text).encode("utf-8")).hexdigest()
                # retain the RAW response content-addressed
                # so a parse vacancy can be replayed by the oracle.
                self._persist_response(row.response_sha256, str(text))
            else:
                row.error = "client returned None"
            self._durable({"event": "call_terminal", "seq": row.seq,
                           "ok": row.ok, "response_sha256": row.response_sha256,
                           "latency_ms": row.latency_ms, "usage": row.usage,
                           "error": row.error})
            return text

        return recording

    def _geno_digest(self, genes: dict) -> str:
        view = getattr(self.adapter, "loci_view", None)
        if callable(view):
            return loci_canonical_digest(view(genes))
        blob = json.dumps(genes, sort_keys=True, ensure_ascii=False,
                          default=_np_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _geno_key(self, genes: dict) -> str:
        """Occupancy key (exact description-and-code match) for Fermi-type selection."""
        view = getattr(self.adapter, "loci_view", None)
        return genotype_key(view(genes)) if callable(view) else self._geno_digest(genes)

    def _op_applied(self, output_digest: str) -> None:
        """Durable operator_applied row: parse-level evidence connecting the
        call to its output genotype digest."""
        with self._io_lock:
            self._op_output[self._ctx["op_id"]] = output_digest
        self._durable({"event": "operator_applied", "seq": 0,
                       "op_id": self._ctx["op_id"], "role": self._ctx["role"],
                       "gen": self._ctx["gen"],
                       "individual_id": self._ctx["individual_id"],
                       "input_digests": list(self._ctx["input_digests"]),
                       "output_digest": output_digest})

    def _alloc_op_id(self) -> str:
        with self._io_lock:
            self._call_seq_ctx = getattr(self, "_call_seq_ctx", 0) + 1
            return f"op{self._call_seq_ctx:05d}"

    def _op_ctx(self, gen: int, role: str, individual_id: "str | None",
                input_digests: "list | None" = None, op_id: "str | None" = None) -> None:
        # a pre-assigned op_id (main thread, canonical order) sets the context without
        # consuming the counter; the legacy path allocates here.
        self._ctx = {"gen": gen, "role": role, "individual_id": individual_id,
                     "op_id": op_id if op_id is not None else self._alloc_op_id(),
                     "input_digests": input_digests or []}

    # ── id + bookkeeping ─────────────────────────────────────────────
    def _new_id(self, prefix: str) -> str:
        self._next += 1
        return f"{prefix}{self._next:04d}"

    def _record(self, ind: Individual) -> Individual:
        self.lineage.append(ind)
        return ind

    def _event(self, gen: int, kind: str, ind_id: str | None, op: str | None,
               parent_ids: list[str], reason: str) -> None:
        self.events.append(Event(gen, kind, ind_id, op, list(parent_ids), reason))

    # ── evaluation (C7: missing is never zero) ───────────────────────
    def _evaluate(self, pool: list[Individual], gen: int) -> tuple[list[Individual], dict[str, np.ndarray]]:
        ok: list[Individual] = []
        embs: dict[str, np.ndarray] = {}
        self._eval_failures: dict[str, str] = {}
        n_events_before = len(self.events)
        # V101 P2: viability checks are pure functions of the genes (sandboxed
        # subprocesses), so they may run concurrently; results are consumed
        # below in POOL ORDER, so events/ledger rows are byte-identical to the
        # sequential run. eval_workers=1 is the sequential path.
        if self._eval_workers > 1 and len(pool) > 1:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=self._eval_workers) as ex:
                results = list(ex.map(lambda ind: viability(self.adapter, ind.genes), pool))
        else:
            results = [viability(self.adapter, ind.genes) for ind in pool]
        for ind, (good, kind, why, e, m) in zip(pool, results):
            if not good:
                self._event(gen, kind, ind.id, ind.op, ind.parent_ids, why)
                continue
            ind.energy = e
            embs[ind.id] = m
            ok.append(ind)
        for ev in self.events[n_events_before:]:
            if ev.individual_id is not None and ev.gen == gen:
                self._eval_failures[ev.individual_id] = f"{ev.kind}: {ev.reason}"
        return ok, embs

    # ── candidate assembly (Steps 3-6) ──────
    def _vacancy(self, gen: int, ind_id: str, origin: str, parent_ids: list,
                 failed_op: str, reason: str) -> None:
        """An intended slot that no candidate filled. Recorded
        as first-class evidence (bound to its failed call rows at export)."""
        self._vacancies.append({"gen": gen, "individual_id": ind_id,
                                "origin": origin, "parent_ids": list(parent_ids),
                                "failed_op": failed_op, "reason": reason})

    def _intended_slots(self) -> int:
        return 2 * self.cfg.n + (0 if self.cfg.parent_policy == "keep_originals_mutate_duplicates" else 1)

    def _build_pool(self, pop: list[Individual], gen: int) -> list[Individual]:
        self._vacancies: list[dict] = []
        keep = self.cfg.parent_policy == "keep_originals_mutate_duplicates"
        union: list[Individual] = []
        if keep:

            # is KEPT as is (no call); bosonic duplicates become mutation
            # targets. No elite carry-over: the elite is one of the originals.
            seen: set[str] = set()
            carry = None
            for p in pop:
                d = self._geno_digest(p.genes)
                if d not in seen:
                    seen.add(d)
                    union.append(self._record(Individual(
                        id=self._new_id("k"), genes=copy.deepcopy(p.genes), gen=gen,
                        op="parent_keep", parent_ids=[p.id], source_id=p.id, host=p.host,
                        meta={"pre_digest": d, "post_digest": d},
                    )))
                else:
                    union.append(self._record(Individual(
                        id=self._new_id("p"), genes=copy.deepcopy(p.genes), gen=gen,
                        op="parent_variant", parent_ids=[p.id], source_id=p.id, host=p.host,
                        meta={"pre_digest": d, "duplicate_of": d},
                    )))
        elif self.cfg.operator_policy == "eoh":
            # EoH-style offspring policy: survivors stay in the pool unchanged (EoH steady-state convention), no
            # calls and no elite copy; the N+1 offspring are generated in _operators_eoh.
            carry = None
            for p in pop:
                d = self._geno_digest(p.genes)
                union.append(self._record(Individual(
                    id=self._new_id("k"), genes=copy.deepcopy(p.genes), gen=gen,
                    op="parent_keep", parent_ids=[p.id], source_id=p.id, host=p.host,
                    meta={"pre_digest": d, "post_digest": d},
                )))
        else:
            elite = min(pop, key=lambda i: i.energy)
            carry = self._record(Individual(
                id=self._new_id("e"), genes=copy.deepcopy(elite.genes), gen=gen,
                op="elite_carryover", parent_ids=[elite.id], source_id=elite.id,
                host=elite.host,
            ))
            # parent variants enter the union unchanged (pre-Step-5); EoH rank allocation draws the N
            # mutation targets by EoH rank allocation instead of one variant per survivor
            sources = (self._rank_alloc(pop, len(pop), "mutate") if self.cfg.parent_alloc == "rank"
                       else [(p, None) for p in pop])
            for p, alloc in sources:
                union.append(self._record(Individual(
                    id=self._new_id("p"), genes=copy.deepcopy(p.genes), gen=gen,
                    op="parent_variant", parent_ids=[p.id], source_id=p.id, host=p.host,
                    meta={"pre_digest": self._geno_digest(p.genes), **({"alloc": alloc} if alloc else {})},
                )))
        if self.cfg.llm_workers >= 1:
            return self._operators_parallel(gen, pop, union, carry)
        order = list(range(len(pop)))
        self.rng.shuffle(order)  # Step 4: disjoint pairing, both orders
        for i in range(0, len(order) - 1, 2):
            a, b = pop[order[i]], pop[order[i + 1]]
            for pa, pb in ((a, b), (b, a)):
                child_id = self._new_id("c")  # destination allocated PRE-call
                self._op_ctx(gen, "cross", child_id,
                             [self._geno_digest(pa.genes), self._geno_digest(pb.genes)])
                g = self.adapter.crossover(
                    copy.deepcopy(pa.genes), copy.deepcopy(pb.genes), self.rng,
                    self._client("operator"))
                if g is None:
                    self._event(gen, "operator_failure", child_id, "cross",
                                [pa.id, pb.id], "crossover returned None")
                    self._vacancy(gen, child_id, "child", [pa.id, pb.id],
                                  "cross", "crossover returned None")
                    continue
                self._op_applied(self._geno_digest(g))
                union.append(self._record(Individual(
                    id=child_id, genes=g, gen=gen, op="cross",
                    parent_ids=[pa.id, pb.id], host=pa.host,
                    meta={"pre_digest": self._geno_digest(g),
                          "op_id": self._ctx["op_id"]},
                )))
        # Step 5 (mutation over the whole union; identity legal, None = vacancy)
        # Step 6 (optional integrity repair via the individual's own host client)
        integrity = (getattr(self.adapter, "integrity", None)
                     if self.cfg.integrity_mode == "full" else None)
        pool = [carry] if carry is not None else []
        for ind in union:
            if ind.op == "parent_keep":
                pool.append(ind)  # original survivor: no operator, no call
                continue
            origin = "child" if ind.op == "cross" else "parent_variant"
            # post-operator stack: canonical for parent variants; for the
            # crossover children it is cfg.child_post_ops (V101 D2 ablations).
            post = self.cfg.child_post_ops if ind.op == "cross" else "mutate+integrity"
            g = ind.genes
            if "mutate" in post:
                self._op_ctx(gen, "mutate", ind.id, [self._geno_digest(g)])
                g = self.adapter.mutate(g, self.cfg.strength, self.rng,
                                        self._client("operator"))
                if g is None:
                    self._event(gen, "operator_failure", ind.id, "mutate",
                                ind.parent_ids, "mutate returned None")
                    self._vacancy(gen, ind.id, origin, ind.parent_ids, "mutate",
                                  "mutate returned None")
                    continue
                self._op_applied(self._geno_digest(g))
            if integrity is not None and "integrity" in post:
                self._op_ctx(gen, "integrity", ind.id, [self._geno_digest(g)])
                g = integrity(g, self.rng, self._client(ind.host))
                if g is None:
                    self._event(gen, "operator_failure", ind.id, "integrity",
                                ind.parent_ids, "integrity returned None")
                    self._vacancy(gen, ind.id, origin, ind.parent_ids,
                                  "integrity", "integrity returned None")
                    continue
                self._op_applied(self._geno_digest(g))
            ind.genes = g
            ind.meta["post_digest"] = self._geno_digest(g)
            pool.append(ind)
        return pool

    # ── EoH parent allocation / operator policy ──
    def _rank_alloc(self, pop: list, k: int, role: str) -> list:
        """EoH parent_selection moved verbatim: population sorted by energy (ties by pool
        order), weights ∝ 1/(rank+1+N), k draws WITH replacement (same pair / same
        individual twice allowed). Main-thread rng in canonical order (batching contract).
        Returns [(individual, alloc_record), ...]."""
        n = len(pop)
        order = sorted(range(n), key=lambda i: (pop[i].energy, i))
        w = [1.0 / (r + 1 + n) for r in range(n)]
        tot = sum(w)
        picks = self.rng.choices(range(n), weights=w, k=k)
        return [(pop[order[r]], {"role": role, "rank": r, "prob": w[r] / tot, "source_id": pop[order[r]].id,
                                 "energy": pop[order[r]].energy}) for r in picks]

    def _operators_eoh(self, gen: int, pop: list, union: list) -> list:
        """EoH-style offspring policy: N+1 offspring by EoH's laws (operator drawn uniformly from e1/e2/m1/m2,
        parents by rank allocation, adapter templates fixed by the draw, no post-ops), then
        the optional Step-6 integrity chain on the offspring only. The kept survivors
        (union, op parent_keep) enter the pool unchanged. Identity fixed in the main
        thread before dispatch, results applied in canonical order (batching contract B)."""
        W = self.cfg.llm_workers
        dg = self._geno_digest
        tasks = []
        for _ in range(len(pop) + 1):
            tpl = self.rng.choice(["e1", "e2", "m1", "m2"])
            if tpl in ("e1", "e2"):
                (pa, aa), (pb, ab) = self._rank_alloc(pop, 2, tpl)
                tasks.append({"kind": "cross", "tpl": tpl, "child_id": self._new_id("c"), "op_id": self._alloc_op_id(),
                              "pa": pa, "pb": pb, "rng": random.Random(self.rng.getrandbits(64)),
                              "digests": [dg(pa.genes), dg(pb.genes)], "alloc": [aa, ab]})
            else:
                (pa, aa), = self._rank_alloc(pop, 1, tpl)
                tasks.append({"kind": "mutate", "tpl": tpl, "child_id": self._new_id("m"), "op_id": self._alloc_op_id(),
                              "pa": pa, "pb": None, "rng": random.Random(self.rng.getrandbits(64)),
                              "digests": [dg(pa.genes)], "alloc": [aa]})

        def run_op(t):
            self._op_ctx(gen, t["kind"], t["child_id"], t["digests"], op_id=t["op_id"])
            if t["kind"] == "cross":
                return self.adapter.crossover(copy.deepcopy(t["pa"].genes), copy.deepcopy(t["pb"].genes),
                                              t["rng"], self._client("operator"), template=t["tpl"])
            return self.adapter.mutate(copy.deepcopy(t["pa"].genes), self.cfg.strength, t["rng"],
                                       self._client("operator"), template=t["tpl"])

        pool = list(union)
        offspring = []
        for t, g in zip(tasks, self._run_tasks(run_op, tasks, W)):
            pids = [t["pa"].id] + ([t["pb"].id] if t["kind"] == "cross" else [])
            self._op_ctx(gen, t["kind"], t["child_id"], t["digests"], op_id=t["op_id"])
            if g is None:
                self._event(gen, "operator_failure", t["child_id"], t["kind"], pids, f"{t['tpl']} returned None")
                self._vacancy(gen, t["child_id"], "child", pids, t["kind"], f"{t['tpl']} returned None")
                continue
            self._op_applied(dg(g))
            offspring.append(self._record(Individual(
                id=t["child_id"], genes=g, gen=gen, op="eoh_" + t["tpl"], parent_ids=pids, host=t["pa"].host,
                meta={"pre_digest": dg(g), "op_id": t["op_id"], "alloc": t["alloc"]})))
        integrity = (getattr(self.adapter, "integrity", None) if self.cfg.integrity_mode == "full" else None)
        if integrity is None:
            for ind in offspring:
                ind.meta["post_digest"] = dg(ind.genes)
                pool.append(ind)
            return pool
        itasks = [{"ind": ind, "op_i": self._alloc_op_id(), "rng": random.Random(self.rng.getrandbits(64))}
                  for ind in offspring]

        def run_i(t):
            ind = t["ind"]
            self._op_ctx(gen, "integrity", ind.id, [dg(ind.genes)], op_id=t["op_i"])
            return integrity(ind.genes, t["rng"], self._client(ind.host))

        for t, g in zip(itasks, self._run_tasks(run_i, itasks, W)):
            ind = t["ind"]
            self._op_ctx(gen, "integrity", ind.id, [dg(ind.genes)], op_id=t["op_i"])
            if g is None:
                self._event(gen, "operator_failure", ind.id, "integrity", ind.parent_ids, "integrity returned None")
                self._vacancy(gen, ind.id, "child", ind.parent_ids, "integrity", "integrity returned None")
                continue
            self._op_applied(dg(g))
            ind.genes = g
            ind.meta["post_digest"] = dg(g)
            pool.append(ind)
        return pool

    # ── batched parallel operator calls ─────────────
    def _run_tasks(self, fn, tasks: list, workers: int) -> list:
        """Run fn over tasks with a bounded pool; results in task order. The first
        exception is detected as soon as it happens (FIRST_EXCEPTION), not-started
        futures are cancelled, running ones are waited for (bounded by the client
        transport timeout/retries), then the exception is re-raised
."""
        from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait  # noqa: PLC0415
        ex = ThreadPoolExecutor(max_workers=max(1, workers))
        futs = []
        try:
            futs = [ex.submit(fn, t) for t in tasks]
            pending = set(futs)
            while pending:
                done, pending = wait(pending, return_when=FIRST_EXCEPTION)
                failed = [f for f in done if f.exception() is not None]
                if failed:
                    for f in pending:
                        f.cancel()
                    raise failed[0].exception()
            return [f.result() for f in futs]
        finally:
            ex.shutdown(wait=True)

    def _operators_parallel(self, gen: int, pop: list, union: list, carry) -> list:
        """Steps 4-6 with the crossover calls and the per-individual
        mutate->integrity chains dispatched to a thread pool. Everything that
        defines identity is fixed in the main thread in canonical order BEFORE
        dispatch: child ids, op ids, per-op rng streams, input digests; results
        are applied in canonical order afterwards, so workers=1..k give the same
        lineage, events, vacancies and digests (contract B of the batching design)."""
        W = self.cfg.llm_workers
        dg = self._geno_digest
        if self.cfg.operator_policy == "eoh":
            return self._operators_eoh(gen, pop, union)
        if self.cfg.parent_alloc == "rank":  # EoH rank allocation: EoH parent allocation, one draw of 2 per call
            pairs = [tuple(self._rank_alloc(pop, 2, "cross")) for _ in range(len(pop))]
        else:
            order = list(range(len(pop)))
            self.rng.shuffle(order)  # Step 4: disjoint pairing, both orders
            pairs = []
            for i in range(0, len(order) - 1, 2):
                a, b = pop[order[i]], pop[order[i + 1]]
                pairs += [((a, None), (b, None)), ((b, None), (a, None))]
        cross_tasks = []
        for (pa, aa), (pb, ab) in pairs:
            cross_tasks.append({"child_id": self._new_id("c"), "op_id": self._alloc_op_id(),
                                "pa": pa, "pb": pb, "rng": random.Random(self.rng.getrandbits(64)),
                                "digests": [dg(pa.genes), dg(pb.genes)],
                                "alloc": None if aa is None else [aa, ab]})

        def run_cross(t):
            self._op_ctx(gen, "cross", t["child_id"], t["digests"], op_id=t["op_id"])
            return self.adapter.crossover(copy.deepcopy(t["pa"].genes), copy.deepcopy(t["pb"].genes),
                                          t["rng"], self._client("operator"))

        for t, g in zip(cross_tasks, self._run_tasks(run_cross, cross_tasks, W)):
            pa, pb, child_id = t["pa"], t["pb"], t["child_id"]
            self._op_ctx(gen, "cross", child_id, t["digests"], op_id=t["op_id"])
            if g is None:
                self._event(gen, "operator_failure", child_id, "cross", [pa.id, pb.id], "crossover returned None")
                self._vacancy(gen, child_id, "child", [pa.id, pb.id], "cross", "crossover returned None")
                continue
            self._op_applied(dg(g))
            union.append(self._record(Individual(
                id=child_id, genes=g, gen=gen, op="cross", parent_ids=[pa.id, pb.id], host=pa.host,
                meta={"pre_digest": dg(g), "op_id": t["op_id"], **({"alloc": t["alloc"]} if t.get("alloc") else {})})))
        # Steps 5-6: one chain per union individual (mutate -> integrity), canonical order
        integrity = (getattr(self.adapter, "integrity", None) if self.cfg.integrity_mode == "full" else None)
        pool = [carry] if carry is not None else []
        chain_tasks = []
        for ind in union:
            if ind.op == "parent_keep":
                chain_tasks.append({"ind": ind, "keep": True})
                continue
            post = self.cfg.child_post_ops if ind.op == "cross" else "mutate+integrity"
            chain_tasks.append({"ind": ind, "keep": False,
                                "origin": "child" if ind.op == "cross" else "parent_variant",
                                "mut": ("mutate" in post), "integ": (integrity is not None and "integrity" in post),
                                "op_m": self._alloc_op_id() if "mutate" in post else None,
                                "op_i": self._alloc_op_id() if (integrity is not None and "integrity" in post) else None,
                                "rng": random.Random(self.rng.getrandbits(64))})

        def run_chain(t):
            if t["keep"]:
                return None
            ind = t["ind"]; g = ind.genes; d_in = dg(g)
            g_m = None
            if t["mut"]:
                self._op_ctx(gen, "mutate", ind.id, [d_in], op_id=t["op_m"])
                g_m = self.adapter.mutate(g, self.cfg.strength, t["rng"], self._client("operator"))
                if g_m is None:
                    return {"stage": "mutate", "g_m": None, "g_i": None}
                g = g_m
            g_i = None
            if t["integ"]:
                self._op_ctx(gen, "integrity", ind.id, [dg(g)], op_id=t["op_i"])
                g_i = integrity(g, t["rng"], self._client(ind.host))
                if g_i is None:
                    return {"stage": "integrity", "g_m": g_m, "g_i": None}
                g = g_i
            return {"stage": "ok", "g_m": g_m, "g_i": g_i, "g": g}

        for t, r in zip(chain_tasks, self._run_tasks(run_chain, chain_tasks, W)):
            ind = t["ind"]
            if t["keep"]:
                pool.append(ind)  # original survivor: no operator, no call
                continue
            d_in = dg(ind.genes)
            if t["mut"] and r["g_m"] is not None:
                self._op_ctx(gen, "mutate", ind.id, [d_in], op_id=t["op_m"])
                self._op_applied(dg(r["g_m"]))
            if r["stage"] == "mutate":
                self._event(gen, "operator_failure", ind.id, "mutate", ind.parent_ids, "mutate returned None")
                self._vacancy(gen, ind.id, t["origin"], ind.parent_ids, "mutate", "mutate returned None")
                continue
            g_after_m = r["g_m"] if r["g_m"] is not None else ind.genes
            if t["integ"] and r["g_i"] is not None:
                self._op_ctx(gen, "integrity", ind.id, [dg(g_after_m)], op_id=t["op_i"])
                self._op_applied(dg(r["g_i"]))
            if r["stage"] == "integrity":
                self._event(gen, "operator_failure", ind.id, "integrity", ind.parent_ids, "integrity returned None")
                self._vacancy(gen, ind.id, t["origin"], ind.parent_ids, "integrity", "integrity returned None")
                continue
            ind.genes = r["g"]
            ind.meta["post_digest"] = dg(r["g"])
            pool.append(ind)
        return pool

    # ── selection + materialisation (C2, C3) ─────────────────────────
    def _select(self, cands: list[Individual], embs: dict[str, np.ndarray],
                gen: int) -> tuple[list[Individual], dict]:
        genotype_of = {c.id: self._geno_key(c.genes) for c in cands}
        result = thermodynamical_select(
            candidate_ids=[c.id for c in cands],
            energies={c.id: c.energy for c in cands},
            embeddings=embs,
            target_size=self.cfg.n,
            num_sections=self.adapter.num_sections,
            dim=self.adapter.dim,
            eps=self.cfg.eps,
            temperature=self.cfg.temperature,
            free_energy_mode=self.cfg.free_energy_mode,
            occupancy=self.cfg.occupancy,
            genotype_of=genotype_of,
        )
        by_id = {c.id: c for c in cands}
        occurrence: dict[str, int] = {}
        survivors: list[Individual] = []
        mapping: list[dict] = []  # C3 instance table
        for cid in result.selected_ids:
            n = occurrence.get(cid, 0)
            occurrence[cid] = n + 1
            src = by_id[cid]
            new_id = self._new_id("s") if n == 0 else self._new_id("s")
            ind = self._record(Individual(
                id=new_id, genes=copy.deepcopy(src.genes), gen=gen,
                op="survivor" if n == 0 else "clone",
                parent_ids=[src.id], source_id=src.id, energy=src.energy,
                host=src.host,
            ))
            mapping.append({"selected_cid": cid, "occurrence": n, "materialized_id": new_id})
            survivors.append(ind)
        if self.cfg.occupancy == "fermion":
            # Safety valve (second layer): the materialised survivors must be one per genotype.
            genos = [self._geno_key(s.genes) for s in survivors]
            srcs = [s.source_id for s in survivors]
            if len(set(genos)) != len(genos) or len(set(srcs)) != len(srcs):
                raise RuntimeError(
                    f"gen{gen}: fermion invariant violated after materialisation "
                    f"(distinct genotypes {len(set(genos))}/{len(genos)}, distinct sources {len(set(srcs))}/{len(srcs)})")
        sel_info = {
            "fermi_audit": {"pool_size": len(cands), "distinct_genotypes_in_pool": len(set(genotype_of.values())),
                            "survivor_distinct_genotypes": len({self._geno_key(s.genes) for s in survivors})},
            "selected_ids": result.selected_ids,
            "instance_mapping": mapping,
            "final_free_energy": result.final_free_energy,
            "final_logdet": result.final_logdet,
            "trace": [asdict(s) for s in result.trace],
        }
        return survivors, sel_info

    def _persist_quarantine(self, gen: int, n_valid: int, n_built: int) -> str:
        """Atomically write partial evidence before aborting."""
        out_dir = Path(self.cfg.out_dir) if self.cfg.out_dir else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"quarantine_{self.adapter.name}_seed{self.cfg.seed}_gen{gen}.json"
        blob = json.dumps({
            "quarantined": True,
            "reason": f"gen{gen}: {n_valid} valid of {n_built} built < N={self.cfg.n}",
            "lineage": [asdict(i) for i in self.lineage],
            "events": [asdict(e) for e in self.events],
            "generation_log": self.generation_log,
            "call_ledger": [asdict(c) for c in self.call_ledger],
            "config": asdict(self.cfg),
            "adapter": self.adapter.name,
        }, indent=1, ensure_ascii=False, default=_np_default)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(path)
        marker = out_dir / f"_QUARANTINED_{self.adapter.name}_seed{self.cfg.seed}.json"
        if not marker.exists():
            marker.write_text(json.dumps({
                "quarantine_file": str(path),
                "quarantine_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                "call_ledger_sha256": self._finalize_ledger(),
            }, indent=1), encoding="utf-8")
        return str(path)

    def _finalize_ledger(self) -> "str | None":
        if self._ledger_fh is None:
            return None if self._ledger_path is None else hashlib.sha256(
                self._ledger_path.read_bytes()).hexdigest()
        self._ledger_fh.close()
        self._ledger_fh = None
        return hashlib.sha256(self._ledger_path.read_bytes()).hexdigest()

    def _atomic_write(self, path: Path, blob: str) -> str:
        """Exclusive-commit atomic write: the destination
        is RESERVED with O_CREAT|O_EXCL (races lose loudly), content lands via
        a uniquely-named temp + replace of our own reservation. Returns the
        content sha256; bytes are written raw so digest == disk."""
        import os as _os
        import uuid as _uuid
        data = blob.encode("utf-8")
        fd = _os.open(str(path), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
        _os.close(fd)  # reservation: any concurrent writer now fails
        tmp = path.with_name(f".{path.name}.{_uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        for attempt in range(5):  # replacing OUR reservation, not third-party work
            try:
                tmp.replace(path)
                break
            except PermissionError:
                # Windows transient lock on the fresh reservation (observed 2026-09-07, x2rank cs20402:
                # WinError 5 on the _UNCERTIFIED marker after a complete run); retry briefly, then fail loudly
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
        return hashlib.sha256(data).hexdigest()

    def _contract_sha(self) -> "str | None":
        spec = Path(__file__).resolve().parents[3] / "docs" / "ALGORITHM.md"
        try:
            return hashlib.sha256(spec.read_bytes()).hexdigest()
        except OSError:
            return None

    def _write_marker(self, kind: str, payload: dict) -> None:
        if not self.cfg.out_dir:
            return
        out = Path(self.cfg.out_dir)
        stem = f"{self.adapter.name}_seed{self.cfg.seed}"
        try:
            self._atomic_write(out / f"{kind}_{stem}.json",
                               json.dumps(payload, indent=1, default=_np_default))
        except FileExistsError:
            pass  # a terminal marker already exists; never overwrite evidence

    def _finalize_terminal(self, result: "RunResult") -> None:
        """Terminal order: ledger closed -> result/trace ->
        Rust validation -> digest-bound _SUCCESS LAST. Anything less than a
        CONFORMANT full-mode run gets _UNCERTIFIED (never a success print)."""
        if not self.cfg.out_dir:
            return
        out = Path(self.cfg.out_dir)
        stem = f"{self.adapter.name}_seed{self.cfg.seed}"
        result_sha = self._atomic_write(out / f"result_{stem}.json", result.to_json())
        base = {
            "adapter": self.adapter.name,
            "config": asdict(self.cfg),
            "config_sha256": hashlib.sha256(json.dumps(
                asdict(self.cfg), sort_keys=True).encode()).hexdigest(),
            "contract_sha256": self._contract_sha(),
            "result_sha256": result_sha,
            "call_ledger_sha256": result.manifest.get("call_ledger_sha256"),
            "git_commit": result.manifest.get("git_commit"),
            "dirty_diff_sha256": result.manifest.get("dirty_diff_sha256"),
            "llm_usage_totals": result.manifest.get("llm_usage_totals"),
            "n_events": len(self.events),
        }
        if (self.cfg.integrity_mode != "full" or self.cfg.child_post_ops != "mutate+integrity"
                or self.cfg.parent_policy != "mutate_all"):
            what = []
            if self.cfg.integrity_mode != "full":
                what.append(f"integrity_mode={self.cfg.integrity_mode}")
            if self.cfg.child_post_ops != "mutate+integrity":
                what.append(f"child_post_ops={self.cfg.child_post_ops}")
            if self.cfg.parent_policy != "mutate_all":
                what.append(f"parent_policy={self.cfg.parent_policy}")
            self._write_marker("_UNCERTIFIED", {
                **base, "reason": ", ".join(what) + " (ablation)"})
            return
        config_json = json.dumps(asdict(self.cfg), sort_keys=True)
        config_sha = hashlib.sha256(config_json.encode()).hexdigest()
        contract_sha = self._contract_sha()
        try:
            from grant_evo.bench.trace_export import to_tgade_trace
            trace = to_tgade_trace(result, self.adapter,
                                   config_sha256=config_sha,
                                   contract_sha256=contract_sha)
        except Exception as exc:  # noqa: BLE001 - reason recorded, never certified
            self._write_marker("_UNCERTIFIED", {**base, "reason": f"trace export: {exc}"})
            return
        oracle = Path(self.cfg.oracle_path) if self.cfg.oracle_path else (
            Path(__file__).resolve().parents[3] / "rust" / "tgade_core" /
            "target" / "release" / "tgade-contract-oracle.exe")
        if not oracle.exists():
            self._write_marker("_UNCERTIFIED", {**base, "reason": "rust oracle unavailable"})
            return
        ledger_jsonl = (self._ledger_path.read_bytes().decode("utf-8")
                        if self._ledger_path else "")  # exact disk bytes
        # raw responses for every parse vacancy (oracle replays the predicate)
        responses: dict = {}
        rdir = out / "responses"
        for generation in trace.get("generations", []):
            for vac in generation.get("vacancies", []):
                if vac.get("failure") != "parse" or not vac.get("events"):
                    continue
                sha = vac["events"][-1].get("response_sha256")
                if sha and (rdir / f"{sha}.txt").exists():
                    responses[sha] = (rdir / f"{sha}.txt").read_text(encoding="utf-8")
        bundle = {"trace": trace, "ledger_jsonl": ledger_jsonl, "responses": responses,
                  "config_json": config_json, "contract_sha256": contract_sha}
        trace_blob = json.dumps({"operation": "validate_bundle", "bundle": bundle},
                                indent=1, default=_np_default)
        trace_sha = self._atomic_write(out / f"trace_{stem}.json", trace_blob)
        report_path = out / f"rust_report_{stem}.json"
        r = subprocess.run([str(oracle), str(out / f"trace_{stem}.json"), str(report_path)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
        if r.returncode != 0:
            self._write_marker("_UNCERTIFIED", {
                **base, "trace_sha256": trace_sha,
                "reason": f"rust validation failed: {r.stderr.strip()[:400]}"})
            return
        report = json.loads(report_path.read_text(encoding="utf-8"))["report"]
        if not report.get("conformant"):
            self._write_marker("_UNCERTIFIED", {
                **base, "trace_sha256": trace_sha, "reason": "not conformant",
                "rust_report": report})
            return
        # re-read verification: every recorded digest
        # must match the bytes on disk BEFORE the success print exists.
        if hashlib.sha256((out / f"result_{stem}.json").read_bytes()).hexdigest() != result_sha:
            self._write_marker("_FAILED", {**base, "error": "result re-read digest mismatch"})
            return
        if hashlib.sha256((out / f"trace_{stem}.json").read_bytes()).hexdigest() != trace_sha:
            self._write_marker("_FAILED", {**base, "error": "trace re-read digest mismatch"})
            return
        if (self._ledger_path is not None and hashlib.sha256(
                self._ledger_path.read_bytes()).hexdigest()
                != result.manifest.get("call_ledger_sha256")):
            self._write_marker("_FAILED", {**base, "error": "ledger re-read digest mismatch"})
            return
        self._write_marker("_SUCCESS", {
            **base,
            "trace_sha256": trace_sha,
            "oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
            "rust_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "rust_report": report,
        })

    # ── main loop ────────────────────────────────────────────────────
    def run(self) -> RunResult:
        try:
            return self._run_inner()
        except PopulationExtinctionError:
            raise  # _QUARANTINED marker + ledger close already persisted
        except BaseException as exc:
            ledger_sha = self._finalize_ledger()
            self._write_marker("_FAILED", {
                "adapter": self.adapter.name, "config": asdict(self.cfg),
                "error": f"{type(exc).__name__}: {exc}",
                "call_ledger_sha256": ledger_sha,
                "n_events": len(self.events),
                "n_lineage": len(self.lineage)})
            raise

    def _run_inner(self) -> RunResult:
        t0 = time.time()
        gen0: list[Individual] = []
        attempts = 0
        if self._gen0_bank is not None:
            # SHARED gen0 (E4): genotypes come from the certified bank; no
            # init calls happen here. Every genotype must match a declared
            # bank digest; the Rust contract re-binds parents to the bank.
            # The bank must cover N exactly - mixing bank and fresh init
            # calls in one run is forbidden (single binding path).
            for g in self._gen0_bank["genotypes"]:
                d = self._geno_digest(g)
                if d not in self._gen0_bank["digests"]:
                    raise ValueError("bank genotype digest not declared")
                gen0.append(self._record(Individual(
                    id=self._new_id("i"), genes=g, gen=0, op="init_bank",
                    host=None)))
        while len(gen0) < self.cfg.n and attempts < 3 * self.cfg.n:
            attempts += 1
            host = self.cfg.hosts[len(gen0) % len(self.cfg.hosts)]
            init_id = self._new_id("i")  # pre-allocated: init evidence is bound
            self._op_ctx(0, "init", init_id)
            g = self.adapter.init_genes(self.rng, self._client(host))
            if g is None:
                self._event(0, "operator_failure", init_id, "init", [], "init returned None")
                continue
            self._op_applied(self._geno_digest(g))  # gen0 state anchor
            gen0.append(self._record(Individual(
                id=init_id, genes=g, gen=0, op="init", host=host)))
        # Replan round 2026-09-06 (A-list): Step-6 repair also covers gen0 so no
        # candidate reaches selection unrepaired. Fail-closed on identity: a
        # repair that changes the loci digest (R-gen on a bank genotype) is
        # skipped with an event, because the bank contract binds digests.
        g0_integrity = (getattr(self.adapter, "integrity", None)
                        if (self.cfg.integrity_mode == "full" and self.cfg.gen0_integrity) else None)
        if g0_integrity is not None:
            for ind in gen0:
                d0 = self._geno_digest(ind.genes)
                self._op_ctx(0, "integrity", ind.id, [d0])
                g2 = g0_integrity(ind.genes, self.rng, self._client(ind.host))
                if g2 is None or self._geno_digest(g2) != d0:
                    self._event(0, "operator_failure", ind.id, "integrity", [],
                                "gen0 integrity skipped: None or loci digest change")
                    continue
                ind.genes = g2
                self._op_applied(d0)
        pop, embs = self._evaluate(gen0, 0)
        if len(pop) < self.cfg.n:
            raise RuntimeError(
                f"gen0: only {len(pop)} valid of {self.cfg.n} after {attempts} attempts")
        pop = pop[: self.cfg.n]
        for gen in range(1, self.cfg.generations + 1):
            parent_ids = [i.id for i in pop]
            parent_digests = {i.id: self._geno_digest(i.genes) for i in pop}
            pool = self._build_pool(pop, gen)
            cands, embs = self._evaluate(pool, gen)
            # Extinction is occupancy-specific.
            # Bosonic occupancy fills N survivor slots from >= 1 valid source
            # (clones are legal), so only ZERO valid candidates is extinction;
            # fermionic occupancy needs N distinct sources.
            min_valid = 1 if self.cfg.occupancy == "boson" else self.cfg.n
            if len(cands) < min_valid:
                path = self._persist_quarantine(gen, len(cands), len(pool))
                raise PopulationExtinctionError(
                    f"gen{gen}: only {len(cands)} valid candidates of {len(pool)} built "
                    f"(< {min_valid} required under {self.cfg.occupancy} occupancy); "
                    f"run quarantined; evidence: {path}")
            pop2, sel_info = self._select(cands, embs, gen)
            pop = pop2
            self.generation_log.append({
                "gen": gen,
                "parent_ids": parent_ids,
                "parent_digests": parent_digests,
                "survivor_digests": {i.id: self._geno_digest(i.genes) for i in pop2},
                "pool_built": len(pool),
                "pool_valid": len(cands),
                # intended slots follow the parent policy: 2N+1 canonical,
                # 2N when originals are kept (no elite slot) --
                # a fixed 2N+1 logged a phantom vacancy per generation.
                "intended_slots": self._intended_slots(),
                "vacancies": self._intended_slots() - len(pool),
                "pool_ids": [i.id for i in pool],
                "vacancy_records": list(self._vacancies),
                "eval_failures": dict(self._eval_failures),
                "best_energy": min(c.energy for c in cands),
                "distinct_survivor_sources": len({i.source_id for i in pop}),
                "clones": sum(1 for i in pop if i.op == "clone"),
                **sel_info,
            })
        manifest = {
            "adapter": self.adapter.name,
            "energy_spec": self.adapter.energy_spec,
            "diversity_spec": self.adapter.diversity_spec,
            "operator_spec": getattr(self.adapter, "operator_spec", None),
            "config": asdict(self.cfg),
            "num_sections": self.adapter.num_sections,
            "dim": self.adapter.dim,
            "elapsed_s": round(time.time() - t0, 3),
            "n_events": len(self.events),
            "llm_calls": len(self.call_ledger),
            "llm_call_failures": sum(1 for c in self.call_ledger if not c.ok),
            "llm_usage_totals": _usage_totals(self.call_ledger),
            "op_output_digests": dict(self._op_output),
            "call_ledger_file": str(self._ledger_path) if self._ledger_path else None,
            "call_ledger_sha256": self._finalize_ledger(),
            **self._git_at_start,
            "git_state_at_end": _git_state(Path(__file__).resolve().parents[3]),
        }
        result = RunResult(pop, self.lineage, self.events, self.generation_log,
                           manifest, call_ledger=self.call_ledger)
        self._finalize_terminal(result)
        return result
