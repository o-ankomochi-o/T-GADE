"""MULTISTART comparator + gen0-bank producer.

MULTISTART = C independent i1 init draws, final selection = argmin TRAIN
energy. Not a generational topology, so the trace contract does not
apply; instead every candidate is bound to durable transport evidence in the
SAME two-event ledger format the engine uses, and
validate_multistart_evidence() enforces ledger completeness:
  - result sha binds the raw ledger bytes;
  - every FRESH candidate's applied row binds 1:1 to a successful
    call_started/call_terminal pair (op_id, role, individual, digests);
  - every BANK candidate's digest is a member of the declared bank
    (bank sha over its canonical digest list);
  - call/attempt counts are exact (quota audit);
  - no surplus applied rows.

The gen0-bank producer is the same machinery with fresh_calls = n and no
bank input: its output file IS the shared gen0 bank for every arm.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import random
import time
from pathlib import Path

from grant_evo.tgade.engine import LLMResult, loci_canonical_digest, viability


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class MultistartRun:
    def __init__(self, adapter, client, *, fresh_calls: int, out_dir: str,
                 seed: int, bank: "dict | None" = None, label: str = "ms",
                 stop_at_valid: "int | None" = None, eval_workers: int = 1):
        # eval_workers (V101 P2): sandbox viability checks of already-generated
        # candidates run concurrently with the following LLM calls. Only in
        # comparator mode (stop_at_valid None): the stop rule needs each
        # verdict before the next call, so producer mode stays sequential.
        # Ledger rows are written at generation time, before viability, so
        # they are byte-identical either way; candidates keep call order.
        self._eval_workers = max(1, int(eval_workers))
        # stop_at_valid: gen0-BANK producer mode - stop once this many valid
        # fresh candidates exist (quota fresh_calls is the hard ceiling, like
        # the engine's 3N init attempts). None = consume the whole quota
        # (MULTISTART comparator semantics, attempts == quota enforced).
        if (isinstance(fresh_calls, bool) or not isinstance(fresh_calls, int)
                or fresh_calls < 0):
            raise ValueError("fresh_calls must be a non-negative integer")
        if stop_at_valid is not None and (isinstance(stop_at_valid, bool)
                                          or not isinstance(stop_at_valid, int)
                                          or stop_at_valid < 1):
            raise ValueError("stop_at_valid must be a positive integer or None")
        self.stop_at_valid = stop_at_valid
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.adapter = adapter
        self.client = client
        self.fresh_calls = int(fresh_calls)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=False)  # no-clobber (S1-4)
        self.seed = seed
        self.bank = bank
        if bank is not None:
            actual = sorted(loci_canonical_digest(adapter.loci_view(g))
                            for g in bank["genotypes"])
            if (actual != sorted(bank["digests"])
                    or bank_digest(bank["digests"]) != bank["sha256"]):
                raise ValueError("bank genotypes and declared digest multiset differ")
        self.label = label
        self._seq = 0
        self._ledger_path = self.out / f"call_ledger_{label}_seed{seed}.jsonl"
        self._fh = open(self._ledger_path, "w", encoding="utf-8", newline="\n")

    def _durable(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def run(self) -> dict:
        rng = random.Random(self.seed)
        candidates = []
        # bank genotypes first (shared gen0; calls live in the BANK's ledger)
        if self.bank is not None:
            for g in self.bank["genotypes"]:
                d = loci_canonical_digest(self.adapter.loci_view(g))
                good, kind, why, e, _m = viability(self.adapter, g)  # P0-7 shared check
                candidates.append({"digest": d, "energy": e if good else None,
                                   "origin": "bank",
                                   "viability": "ok" if good else f"{kind}: {why}"})
        attempts = 0
        pipeline = self.stop_at_valid is None and self._eval_workers > 1
        pool = pending = None
        if pipeline:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            pool = ThreadPoolExecutor(max_workers=self._eval_workers)
            pending = []  # (candidate placeholder, future) in call order
        try:
            attempts, candidates = self._fresh_loop(rng, candidates, pipeline, pool, pending)
        except BaseException:
            # on any failure cancel queued evaluations, drain
            # the running ones and close the ledger before propagating.
            if pool is not None:
                for _c, fut in pending:
                    fut.cancel()
                pool.shutdown(wait=True)
            self._fh.close()
            raise
        if pipeline:
            pool.shutdown(wait=True)
        self._fh.close()
        ledger_sha = _sha(self._ledger_path.read_bytes())
        valid = [c for c in candidates if c["energy"] is not None]
        best = min(valid, key=lambda c: c["energy"]) if valid else None
        return self._finish(candidates, attempts, ledger_sha, valid, best)

    def _fresh_loop(self, rng, candidates, pipeline, pool, pending):
        attempts = 0
        for i in range(self.fresh_calls):
            if self.stop_at_valid is not None and sum(
                    1 for c in candidates
                    if c["origin"] == "fresh" and c["energy"] is not None
            ) >= self.stop_at_valid:
                break
            op_id = f"ms{i:05d}"
            ind = f"m{i:05d}"

            def recording(prompt: str, **opts):
                self._seq += 1
                self._durable({"event": "call_started", "ts": time.time(),
                               "seq": self._seq, "op_id": op_id, "gen": 0,
                               "role": "init", "individual_id": ind,
                               "requested_host": None, "resolved_host": None,
                               "prompt_sha256": _sha(str(prompt).encode("utf-8")),
                               "input_digests": [], "opts": dict(opts)})
                try:
                    out = self.client(prompt, **opts) if opts else self.client(prompt)
                except Exception as exc:
                    self._durable({"event": "call_terminal", "seq": self._seq,
                                   "ok": False,
                                   "error": f"{type(exc).__name__}: {exc}",
                                   # cost of earlier physical attempts of this
                                   # logical call
                                   "usage": dict(getattr(exc, "usage", None) or {})})
                    raise
                text = out.text if isinstance(out, LLMResult) else out
                usage = dict(out.usage) if isinstance(out, LLMResult) else {}
                sha = _sha(str(text).encode("utf-8")) if text is not None else None
                if sha is not None:  # content-addressed raw response (P0-1)
                    rdir = self.out / "responses"
                    rdir.mkdir(exist_ok=True)
                    p = rdir / f"{sha}.txt"
                    if not p.exists():
                        tmp = p.with_suffix(".tmp")
                        tmp.write_text(str(text), encoding="utf-8", newline="\n")
                        tmp.replace(p)
                self._durable({"event": "call_terminal", "seq": self._seq,
                               "ok": text is not None, "response_sha256": sha,
                               "usage": usage})  # per-arm resource evidence (P0-8)
                return text

            attempts += 1
            g = self.adapter.init_genes(rng, recording)
            if g is None:
                continue  # failed extraction: call is charged, no candidate
            d = loci_canonical_digest(self.adapter.loci_view(g))
            self._durable({"event": "operator_applied", "seq": 0,
                           "op_id": op_id, "role": "init", "gen": 0,
                           "individual_id": ind, "input_digests": [],
                           "output_digest": d})
            cand = {"digest": d, "energy": None, "origin": "fresh", "op_id": op_id,
                    "individual_id": ind, "genes": g, "viability": None}
            candidates.append(cand)
            if pipeline:
                pending.append((cand, pool.submit(viability, self.adapter, g)))
            else:
                good, kind, why, e, _m = viability(self.adapter, g)  # P0-7 shared check
                cand["energy"] = e if good else None
                cand["viability"] = "ok" if good else f"{kind}: {why}"
        if pipeline:
            for cand, fut in pending:  # call order, same as the sequential path
                good, kind, why, e, _m = fut.result()
                cand["energy"] = e if good else None
                cand["viability"] = "ok" if good else f"{kind}: {why}"
        return attempts, candidates

    def _finish(self, candidates, attempts, ledger_sha, valid, best) -> dict:
        result = {
            "label": self.label, "seed": self.seed,
            "fresh_calls_quota": self.fresh_calls, "attempts": attempts,
            "stop_at_valid": self.stop_at_valid,
            "bank_sha256": self.bank["sha256"] if self.bank else None,
            "n_bank": len(self.bank["genotypes"]) if self.bank else 0,
            # genes are PERSISTED for fresh candidates ( the
            # selected program must be replayable from disk evidence)
            "candidates": list(candidates),
            "best_digest": best["digest"] if best else None,
            "best_energy": best["energy"] if best else None,
            "call_ledger_sha256": ledger_sha,
        }
        blob = json.dumps(result, indent=1)
        (self.out / f"result_{self.label}_seed{self.seed}.json").write_bytes(
            blob.encode("utf-8"))
        (self.out / f"_MS_COMPLETE_{self.label}_seed{self.seed}.json").write_bytes(
            json.dumps({"result_sha256": _sha(blob.encode("utf-8")),
                        "call_ledger_sha256": ledger_sha}).encode("utf-8"))
        result["_genotypes"] = [c.get("genes") for c in candidates]
        return result


def bank_digest(digests) -> str:
    """Canonical bank identity: sha256 of newline-joined sorted digests
    (+ trailing newline). Identical construction in Python and Rust."""
    joined = chr(10).join(sorted(digests)) + chr(10)
    return _sha(joined.encode("utf-8"))


def bank_from_result(result: dict, genotypes: list) -> dict:
    """Package a producer run's output as the shared gen0 bank."""
    source_genotypes = result.get("_genotypes")
    if (not isinstance(source_genotypes, list)
            or len(source_genotypes) != len(result["candidates"])):
        raise ValueError("producer result lacks genotype/candidate alignment")
    available = [(g, c["digest"]) for g, c in
                 zip(source_genotypes, result["candidates"], strict=True)
                 if (g is not None and c["origin"] == "fresh"
                     and c["energy"] is not None)]
    digests = []
    for chosen in genotypes:
        match = next((i for i, (g, _) in enumerate(available)
                      if g == chosen), None)
        if match is None:
            raise ValueError("bank genotype is not backed by producer result")
        _, digest = available.pop(match)
        digests.append(digest)
    digests.sort()
    return {"sha256": bank_digest(digests), "digests": digests,
            "genotypes": genotypes,
            "source_ledger_sha256": result["call_ledger_sha256"]}


def validate_multistart_evidence(out_dir: "str | Path", label: str,
                                 seed: int, bank: "dict | None" = None,
                                 expected_fresh_calls: "int | None" = None) -> None:
    """Ledger-completeness validation; raises ValueError on any forgery."""
    out = Path(out_dir)
    result_path = out / f"result_{label}_seed{seed}.json"
    marker = json.loads((out / f"_MS_COMPLETE_{label}_seed{seed}.json")
                        .read_bytes())
    raw = result_path.read_bytes()
    if _sha(raw) != marker["result_sha256"]:
        raise ValueError("result bytes do not match the completion marker")
    result = json.loads(raw)
    if result.get("label") != label or result.get("seed") != seed:
        raise ValueError("result label/seed does not match requested run")
    if (expected_fresh_calls is not None
            and result.get("fresh_calls_quota") != expected_fresh_calls):
        raise ValueError("fresh-call quota does not match the declared quota")
    ledger_raw = (out / f"call_ledger_{label}_seed{seed}.jsonl").read_bytes()
    if _sha(ledger_raw) != result["call_ledger_sha256"]:
        raise ValueError("ledger bytes do not match the recorded digest")
    if marker.get("call_ledger_sha256") != result["call_ledger_sha256"]:
        raise ValueError("completion marker does not bind the ledger digest")
    started, terminal, applied = {}, {}, {}
    for line in ledger_raw.decode("utf-8").splitlines():
        row = json.loads(line)
        ev = row["event"]
        if ev == "call_started":
            if row["seq"] in started:
                raise ValueError("duplicate call_started")
            started[row["seq"]] = row
        elif ev == "call_terminal":
            if row["seq"] in terminal:
                raise ValueError("duplicate call_terminal")
            terminal[row["seq"]] = row
        elif ev == "operator_applied":
            if row["op_id"] in applied:
                raise ValueError("duplicate operator_applied")
            applied[row["op_id"]] = row
        else:
            raise ValueError("unknown ledger event")
    if set(started) != set(terminal):
        raise ValueError("incomplete call triple")
    if len(started) != result["attempts"]:
        raise ValueError("attempt count does not match the ledger")
    stop_at_valid = result.get("stop_at_valid")
    if stop_at_valid is None:
        if result["attempts"] != result["fresh_calls_quota"]:
            raise ValueError("call count does not equal the fixed quota")
    else:
        # bank-producer semantics: stop exactly when stop_at_valid valid fresh
        # candidates exist, or exhaust the quota; never both short.
        n_valid_fresh = sum(1 for c in result["candidates"]
                            if c.get("origin") == "fresh" and c.get("energy") is not None)
        if result["attempts"] > result["fresh_calls_quota"]:
            raise ValueError("call count exceeds the fixed quota")
        if not (n_valid_fresh == stop_at_valid
                or (n_valid_fresh < stop_at_valid
                    and result["attempts"] == result["fresh_calls_quota"])):
            raise ValueError("bank producer stopped neither at stop_at_valid nor at quota")
    origins = {c.get("origin") for c in result["candidates"]}
    if not origins <= {"fresh", "bank"}:
        raise ValueError("unknown candidate origin")
    fresh = [c for c in result["candidates"] if c["origin"] == "fresh"]
    for c in fresh:
        ap = applied.get(c["op_id"])
        if (ap is None or ap["individual_id"] != c["individual_id"]
                or ap["output_digest"] != c["digest"] or ap["role"] != "init"):
            raise ValueError("candidate not bound to an applied row")
        backed = any(s["op_id"] == c["op_id"]
                     and s["individual_id"] == c["individual_id"]
                     and terminal[s["seq"]].get("ok")
                     for s in started.values())
        if not backed:
            raise ValueError("applied row without a successful bound call")
    consumed = {c["op_id"] for c in fresh}
    if set(applied) - consumed:
        raise ValueError("surplus operator_applied row")
    bank_cands = [c for c in result["candidates"] if c["origin"] == "bank"]
    if bank is None:
        if (bank_cands or result["bank_sha256"] is not None
                or result.get("n_bank") != 0):
            raise ValueError("bank candidates present but no bank declared")
    else:
        if (bank_digest(bank["digests"]) != bank["sha256"]
                or result["bank_sha256"] != bank["sha256"]):
            raise ValueError("bank sha mismatch")
        if (sorted(c["digest"] for c in bank_cands) != sorted(bank["digests"])
                or result.get("n_bank") != len(bank["digests"])):
            raise ValueError("bank candidate multiset does not match declared bank")

    valid = []
    for c in result["candidates"]:
        energy = c.get("energy")
        if energy is None:
            continue
        if (isinstance(energy, bool) or not isinstance(energy, numbers.Real)
                or not math.isfinite(float(energy))):
            raise ValueError("candidate energy must be finite or null")
        valid.append(c)
    expected_best = min(valid, key=lambda c: c["energy"]) if valid else None
    if expected_best is None:
        if result.get("best_digest") is not None or result.get("best_energy") is not None:
            raise ValueError("best result exists without a valid candidate")
    elif (result.get("best_digest") != expected_best["digest"]
          or result.get("best_energy") != expected_best["energy"]):
        raise ValueError("best result is inconsistent with candidate energies")
