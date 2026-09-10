"""RunResult -> TgadeTrace exporter (G2b S1-5).

Produces the canonical evidence trace consumed by the Rust contract oracle
(`{"operation": "validate_trace", "trace": ...}`). The trace is only emitted
for CONFORMANT-shaped runs: any operator vacancy, missing candidate, or
evaluation exclusion makes the run non-canonical and the exporter refuses
(fail-closed) — a non-conformant run must be argued from the quarantine
evidence, not from a doctored trace.

The adapter must provide ``loci_view(genes) -> dict[str, str]`` mapping the
genotype onto >= 2 named semantic loci with non-empty text (RawSemanticGenome
contract).
"""

from __future__ import annotations

from grant_evo.tgade.engine import RunResult, loci_canonical_digest

_KIND = {"mutate": "mutation", "integrity": "integrity_repair",
         "cross": "semantic_crossover"}


class TraceExportError(RuntimeError):
    pass


_OCCUPANCY = {"boson": "bosonic", "fermion": "fermionic"}


def to_tgade_trace(result: RunResult, adapter, *, config_sha256: "str | None" = None,
                   contract_sha256: "str | None" = None) -> dict:
    loci_view = getattr(adapter, "loci_view", None)
    if loci_view is None:
        raise TraceExportError(
            f"adapter {adapter.name!r} defines no loci_view(genes)")
    by_id = {i.id: i for i in result.lineage}
    # the canonical contract knows only the 2N+1
    # pool (elite / parent_variant / cross). Refuse the parent-keep variant
    # explicitly instead of exporting a partial trace with dangling refs.
    policy = (result.manifest.get("config") or {}).get("parent_policy", "mutate_all")
    if policy != "mutate_all" or any(i.op == "parent_keep" for i in result.lineage):
        raise TraceExportError(
            f"parent_policy={policy!r} is a declared variant outside the canonical "
            "2N+1 contract; no trace export (run stays _UNCERTIFIED)")
    sample = loci_view(result.population[0].genes)
    expected_loci = sorted(sample)
    if len(expected_loci) < 2:
        raise TraceExportError("loci_view must expose >= 2 loci")

    calls_by_ind: dict[str, list] = {}
    for c in result.call_ledger:
        if c.individual_id is not None:
            calls_by_ind.setdefault(c.individual_id, []).append(c)

    generations = []
    for log in result.generation_log:
        gen = log["gen"]
        if "parent_ids" not in log or "pool_ids" not in log:
            raise TraceExportError(
                f"gen{gen}: missing parent_ids/pool_ids (run is not canonical)")
        pool_ids = set(log["pool_ids"])
        eval_failures = dict(log.get("eval_failures", {}))
        candidates = []
        for ind in result.lineage:
            if ind.gen != gen or ind.id not in pool_ids or ind.op not in (
                    "elite_carryover", "parent_variant", "cross"):
                continue
            op_out = getattr(result, "_op_output", None) or result.manifest.get(
                "op_output_digests", {})
            events = [
                {"kind": _KIND[c.role], "uses_llm": True, "success": c.ok,
                 "op_id": c.op_id, "prompt_sha256": c.prompt_sha256,
                 "response_sha256": c.response_sha256,
                 "seq": c.seq, "gen": c.gen, "individual_id": c.individual_id,
                 "input_genotype_sha256": list(c.input_digests),
                 "output_genotype_sha256": op_out.get(c.op_id)}
                for c in calls_by_ind.get(ind.id, [])
                if c.role in _KIND
            ]
            origin = {"elite_carryover": "elite",
                      "parent_variant": "parent_variant",
                      "cross": "child"}[ind.op]
            candidates.append({
                "individual_id": ind.id,
                "source_id": ind.id,
                "origin": origin,
                "parent_ids": list(ind.parent_ids),
                "genome": {"loci": loci_view(ind.genes)},
                "events": events,
                "genotype_sha256": loci_canonical_digest(loci_view(ind.genes)),
                "evaluation_failure": eval_failures.get(ind.id),
            })
        # Intended slots that no candidate filled, bound to their
        # failed call rows. failure = transport (terminal ok=false) or parse
        # (terminal ok=true but no applied genotype).
        vacancies = []
        op_out = getattr(result, "_op_output", None) or result.manifest.get(
            "op_output_digests", {})
        for v in log.get("vacancy_records", []):
            calls = [c for c in calls_by_ind.get(v["individual_id"], [])
                     if c.role in _KIND and c.gen == gen]
            if not calls:
                raise TraceExportError(
                    f"gen{gen}: vacancy {v['individual_id']} has no call evidence")
            events = [
                {"kind": _KIND[c.role], "uses_llm": True, "success": c.ok,
                 "op_id": c.op_id, "prompt_sha256": c.prompt_sha256,
                 "response_sha256": c.response_sha256,
                 "seq": c.seq, "gen": c.gen, "individual_id": c.individual_id,
                 "input_genotype_sha256": list(c.input_digests),
                 "output_genotype_sha256": op_out.get(c.op_id)}
                for c in calls
            ]
            last = calls[-1]
            vacancies.append({
                "individual_id": v["individual_id"],
                "origin": v["origin"],
                "parent_ids": list(v["parent_ids"]),
                "failed_kind": _KIND[v["failed_op"]],
                "failure": "parse" if last.ok else "transport",
                "events": events,
                "reason": v.get("reason"),
            })
        survivors = []
        steps = log["trace"]
        for m, step in zip(log["instance_mapping"], steps):
            survivors.append({
                "individual_id": m["materialized_id"],
                "source_id": m["selected_cid"],
                "delta_f": step["delta_f"],
            })
        cfg_occ = result.manifest.get("config", {}).get("occupancy")
        if cfg_occ not in _OCCUPANCY:
            raise TraceExportError(f"unknown occupancy {cfg_occ!r} (transcription refused)")
        generations.append({
            "generation": gen - 1,  # oracle ordinals are 0-based contiguous
            "parent_ids": list(log["parent_ids"]),
            "parent_digests": dict(log["parent_digests"]),
            "survivor_digests": dict(log["survivor_digests"]),
            "candidates": candidates,
            "vacancies": vacancies,
            "survivors": survivors,
            "occupancy": _OCCUPANCY[cfg_occ],
        })
    cfg = result.manifest.get("config", {})
    return {
        "expected_loci": expected_loci,
        "generations": generations,
        "integrity_mode": cfg.get("integrity_mode"),
        "ledger_sha256": result.manifest.get("call_ledger_sha256"),
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
    }
