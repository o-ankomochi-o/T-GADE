"""Report gate for the paid campaign runner.

ONE fail-closed validator that trusts NOTHING self-reported: every number
the frozen parameter table relies on is recomputed from raw evidence.

  - report bytes == seal (shared convention) and == the freeze-named digest;
  - schema, finite values, engine_certified, diagnostics_complete;
  - execution conditions (model, retries, interval, provider pinning, price
    ceilings, token maxima) == the frozen table in bench.e4_params;
  - diag ledger: digest, completeness, counts, nested physical attempts
    (count == retries+1, charges sum == cost, last attempt == terminal
    status, provider pinned), retained raw responses digest-checked;
  - per-attempt rows: each row is BOUND to a ledger seq and terminal
    response digest; the genotype is RECONSTRUCTED from the retained raw
    response with the adapter's pure parser; canonical digest, energy,
    diversity, dE, and beat flags are RECOMPUTED with the adapter (sandboxed
    for bp_online) and compared; a row claiming failure while the response
    reconstructs to a viable genotype is a hidden success (rejected);
  - every M2 / O-gate summary and attempts_exact recomputed from the
    RECOMPUTED values (never from row values);
  - the declared O2/O3 stop rule; T* rule; source-run digest chain;
  - provenance: git commit (== freeze commit when given), bench paths clean.

Computer-science machinery only; evolutionary vocabulary is borrowed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from grant_evo.bench.diagnostics import _entropy_of_rows, _summary, calibrated_temperature
from grant_evo.bench.seal import check_seal, sha256_bytes
from grant_evo.tgade.engine import loci_canonical_digest, viability


class E3GateError(RuntimeError):
    pass


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise E3GateError(msg)


def _close(a, b, tol=1e-6) -> bool:
    return _finite(a) and _finite(b) and abs(a - b) <= tol


def _same(a, b, tol=1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k], tol) for k in a)
    if _finite(a) and _finite(b):
        return abs(a - b) <= tol
    return a == b


def _o_unsuitable(paired_rate, arm_summary, o1_summary) -> bool:
    if paired_rate is None or not arm_summary or not o1_summary:
        return True
    mean, se, o1m = arm_summary.get("mean"), arm_summary.get("se"), o1_summary.get("mean")
    if not (_finite(mean) and _finite(o1m)):
        return True
    se = se if _finite(se) else 0.0
    return paired_rate < 0.10 and (mean - 2.0 * se) > o1m


def _build_adapter(design: dict):
    name = design.get("adapter")
    cfg = design.get("adapter_config") or {}
    if name == "mock":
        from grant_evo.bench.mock import MockProblem
        return MockProblem()
    if name == "bp_online":
        from grant_evo.bench.adapters.bp_online import BpOnlineAdapter
        return BpOnlineAdapter(**cfg)
    raise E3GateError(f"unknown adapter {name!r} in report design")


def _recompute_rows(rows: list[dict], adapter, terminal: dict, responses_dir: Path,
                    population_by_digest: dict) -> list[dict]:
    """Return rows with RECOMPUTED fields (r_ok, r_energy, r_div, r_dE,
    r_beat) after binding each row to the ledger and reconstructing the
    genotype from the raw response."""
    genes_by_phase_attempt: dict = {}
    energy_cache: dict = {}

    def energy_of(genes):
        d = loci_canonical_digest(adapter.loci_view(genes))
        if d not in energy_cache:
            ok, _k, _w, e, m = viability(adapter, genes)
            energy_cache[d] = (ok, e, m)
        return energy_cache[d]

    out = []
    for row in rows:
        seq = row.get("seq")
        _require(seq is not None and seq in terminal, "diag row not bound to a ledger call")
        term = terminal[seq]
        _require(row.get("response_sha256") == term.get("response_sha256"),
                 "diag row response digest differs from its ledger terminal row")
        text = None
        if term.get("ok") and term.get("response_sha256"):
            p = responses_dir / f"{term['response_sha256']}.txt"
            _require(p.exists(), "retained raw response missing for a diag call")
            raw = p.read_bytes()
            _require(sha256_bytes(raw) == term["response_sha256"],
                     "retained raw response digest mismatch")
            text = raw.decode("utf-8")
        phase = row.get("phase")
        if phase == "m2":
            parent = population_by_digest.get(row.get("parent_digest"))
            _require(parent is not None, "M2 parent digest is not in the source population")
            role, parents, strength = "mutate", [parent], row.get("strength")
        elif phase in ("o1", "o2_init", "o3_init"):
            role, parents, strength = "init", [], "mid"
        elif phase == "o2_cross":
            pa = genes_by_phase_attempt.get(("o2_init", row["parent_attempts"][0]))
            pb = genes_by_phase_attempt.get(("o2_init", row["parent_attempts"][1]))
            _require(pa is not None and pb is not None, "crossover parents not reconstructed")
            role, parents, strength = "cross", [pa, pb], "mid"
        elif phase == "o3_mut":
            pa = genes_by_phase_attempt.get(("o3_init", row["parent_attempt"]))
            _require(pa is not None, "mutation parent not reconstructed")
            role, parents, strength = "mutate", [pa], row.get("strength", "mid")
        else:
            raise E3GateError(f"unknown diag row phase {phase!r}")
        recon = adapter.reconstruct(role, text, parents, strength) if text is not None else None
        r_ok, r_energy, r_div = False, None, None
        if recon is not None:
            ok, e, m = energy_of(recon)
            if ok:
                r_ok, r_energy, r_div = True, e, np.asarray(m, dtype=float).reshape(-1)
        # a row that claims failure while the response reconstructs to a
        # viable genotype hides a success; a row that claims success must
        # match the reconstruction exactly.
        _require(bool(row.get("ok")) == r_ok, "diag row success flag contradicts the raw response")
        rr = dict(row)
        rr.update({"r_ok": r_ok})
        if r_ok:
            _require(loci_canonical_digest(adapter.loci_view(recon))
                     == (row.get("child_digest") or row.get("digest")),
                     "diag row genotype digest differs from the reconstructed genotype")
            _require(_close(row.get("energy"), r_energy, 1e-9),
                     "diag row energy differs from the recomputed energy")
            _require(row.get("div") is not None
                     and np.allclose(np.asarray(row.get("div"), dtype=float), r_div, atol=1e-9),
                     "diag row diversity differs from the recomputed diversity")
            rr.update({"r_energy": r_energy, "r_div": r_div})
            if phase in ("o1", "o2_init", "o3_init"):
                genes_by_phase_attempt[(phase, row["attempt"])] = recon
            if phase == "m2":
                pok, pe, _pm = energy_of(parents[0])
                _require(pok and _close(row.get("parent_energy"), pe, 1e-9),
                         "M2 parent energy differs from the recomputed value")
                rr["r_dE"] = r_energy - pe
                _require(_close(row.get("dE"), rr["r_dE"], 1e-9), "M2 dE differs")
            if phase == "o2_cross":
                pe = [energy_of(p)[1] for p in parents]
                rr["r_beat"] = bool(r_energy < min(pe) - 1e-12)
                _require(bool(row.get("beat_best_parent")) == rr["r_beat"],
                         "crossover beat flag differs from the recomputed value")
            if phase == "o3_mut":
                pe = energy_of(parents[0])[1]
                rr["r_beat"] = bool(r_energy < pe - 1e-12)
                _require(bool(row.get("beat_parent")) == rr["r_beat"],
                         "mutation beat flag differs from the recomputed value")
        out.append(rr)
    return out


def _summaries(rows: list[dict], strengths=("weak", "mid", "strong")) -> dict:
    """Summaries from RECOMPUTED values only."""
    m2 = {}
    for s in strengths:
        rs = [r for r in rows if r.get("phase") == "m2" and r.get("strength") == s]
        ok = [r for r in rs if r["r_ok"]]
        des = [r["r_dE"] for r in ok]
        m2[s] = {
            "n_children": len(ok), "failures": len(rs) - len(ok),
            "child_diversity_entropy": round(_entropy_of_rows(
                [r["r_div"].reshape(1, -1) for r in ok]), 4),
            "dE_mean": round(float(np.mean(des)), 5) if des else None,
            "dE_std": round(float(np.std(des)), 5) if des else None,
            "dE_p_improve": round(float(np.mean([d < 0 for d in des])), 3) if des else None,
        }
    o1 = [r for r in rows if r.get("phase") == "o1"]
    e1 = [r["r_energy"] for r in o1 if r["r_ok"]]
    o2i = [r for r in rows if r.get("phase") == "o2_init"]
    o2c = [r for r in rows if r.get("phase") == "o2_cross"]
    o3i = [r for r in rows if r.get("phase") == "o3_init"]
    o3m = [r for r in rows if r.get("phase") == "o3_mut"]
    half = [r["r_energy"] for r in o2i if r["r_ok"]]
    kids = [r["r_energy"] for r in o2c if r["r_ok"]]
    paired = [r["r_beat"] for r in o2c if r["r_ok"]]
    half3 = [r["r_energy"] for r in o3i if r["r_ok"]]
    kids3 = [r["r_energy"] for r in o3m if r["r_ok"]]
    paired3 = [r["r_beat"] for r in o3m if r["r_ok"]]
    return {"m2": m2, "o_gates": {
        "O1": {**(_summary(e1) if e1 else {"n": 0}), "attempts": len(o1),
               "failures": len(o1) - len(e1)},
        "O2": {"paired_beat_best_parent": round(float(np.mean(paired)), 3) if paired else None,
               "n_children": len(kids), "failures": len(o2c) - len(kids),
               "init_attempts": len(o2i), "init_failures": len(o2i) - len(half),
               "init_valid": len(half), "cross_attempts": len(o2c),
               "arm_utility": {"o1_same_budget": _summary(e1) if e1 else None,
                               "half_plus_children": _summary(half + kids) if kids else None}},
        "O3": {"paired_beat_parent": round(float(np.mean(paired3)), 3) if paired3 else None,
               "n_children": len(kids3), "failures": len(o3m) - len(kids3),
               "init_attempts": len(o3i), "init_failures": len(o3i) - len(half3),
               "init_valid": len(half3), "mutation_attempts": len(o3m),
               "arm_utility": {"o1_same_budget": _summary(e1) if e1 else None,
                               "half_plus_mutants": _summary(half3 + kids3) if kids3 else None}},
    }}


def validate_e3_report(report_path: "str | Path", *, expected_sha: "str | None" = None,
                       expected_commit: "str | None" = None,
                       require_clean: bool = True, n_diag: int = 8) -> dict:
    rp = Path(report_path)
    d = rp.parent
    raw = rp.read_bytes()
    sha = sha256_bytes(raw)
    _require(check_seal(rp), "report seal missing or mismatched")
    if expected_sha is not None:
        _require(sha == expected_sha, "report digest differs from the freeze-named digest")
    rep = json.loads(raw)
    for key in ("design", "m2", "m3", "t_star_e4_rule", "o_gates", "diag_ledger_sha256",
                "diag_rows_sha256", "diagnostic_calls", "engine_certified",
                "diagnostics_complete", "source_run", "git_commit"):
        _require(key in rep, f"report lacks {key}")
    _require(rep["engine_certified"] is True, "engine not certified")
    _require(rep["diagnostics_complete"] is True, "diagnostics not complete")
    _require(rep.get("m2_m3_error") is None, "report carries a diagnostics error")
    if expected_commit is not None:
        _require(rep["git_commit"] == expected_commit, "report git commit differs from freeze")
    if require_clean:
        _require(rep.get("bench_dirty_diff_sha256") in (None, ""),
                 "report generated from a dirty bench tree")
        _require(bool(rep.get("git_commit")), "report lacks a git commit")
    design = rep["design"]
    paid = design.get("model") is not None
    if paid:
        from grant_evo.bench.e4_params import execution_conditions
        _require(design.get("execution") == execution_conditions(),
                 "E3 execution conditions differ from the frozen table")

    # ── diag ledger ───────────────────────────────────────────────────
    ledger_raw = (d / "diag_ledger.jsonl").read_bytes()
    _require(sha256_bytes(ledger_raw) == rep["diag_ledger_sha256"], "diag ledger digest mismatch")
    started, terminal = {}, {}
    for line in ledger_raw.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "call_started":
            _require(row["seq"] not in started, "duplicate diag call_started")
            started[row["seq"]] = row
        elif row.get("event") == "call_terminal":
            _require(row["seq"] not in terminal, "duplicate diag call_terminal")
            terminal[row["seq"]] = row
        else:
            raise E3GateError("unknown diag ledger event")
    _require(set(started) == set(terminal), "diag ledger incomplete (dangling call)")
    dc = rep["diagnostic_calls"]
    fails = sum(1 for r in terminal.values() if not r.get("ok"))
    _require(dc.get("started") == len(started) and dc.get("completed") == len(terminal)
             and dc.get("failures") == fails, "diagnostic_calls do not match the ledger")
    for r in terminal.values():
        u = r.get("usage") or {}
        attempts = u.get("attempts")
        if attempts is not None:  # nested physical attempts
            _require(len(attempts) == int(u.get("retries", 0)) + 1,
                     "physical attempt count differs from retries+1")
            _require(_close(sum(float(a.get("charged_usd", 0.0)) for a in attempts),
                            float(u.get("cost_usd", 0.0)), 1e-9),
                     "physical attempt charges do not sum to the call cost")
            _require(bool(attempts[-1].get("ok")) == bool(r.get("ok")),
                     "last physical attempt status differs from the terminal status")
            if paid and not design["execution"]["allow_fallbacks"]:
                for a in attempts:
                    if a.get("provider") is not None:
                        _require(a["provider"] in design["execution"]["provider_order"],
                                 "a physical attempt was served by a non-pinned provider")
        if r.get("ok") and r.get("response_sha256"):
            p = d / "responses" / f"{r['response_sha256']}.txt"
            _require(p.exists() and sha256_bytes(p.read_bytes()) == r["response_sha256"],
                     "diagnostic raw response missing or digest mismatch")

    # ── rows: binding + reconstruction + recomputation ───────────────
    rows_raw = (d / "diag_rows.jsonl").read_bytes()
    _require(sha256_bytes(rows_raw) == rep["diag_rows_sha256"], "diag rows digest mismatch")
    rows = [json.loads(ln) for ln in rows_raw.decode("utf-8").splitlines() if ln]
    seqs = [r.get("seq") for r in rows]
    _require(len(seqs) == len(set(seqs)), "two diag rows bound to the same ledger call")
    m2_calls = sum(1 for r in started.values() if r.get("phase") == "m2")
    o_calls = sum(1 for r in started.values() if r.get("phase") == "o_gates")
    _require(m2_calls == sum(1 for r in rows if r.get("phase") == "m2"),
             "M2 call count differs from M2 attempt rows")
    _require(o_calls == sum(1 for r in rows if str(r.get("phase", "")).startswith("o")),
             "O-gate call count differs from O-gate attempt rows")

    # source-run digest chain + population for parent reconstruction
    src = rep["source_run"]
    if src is None:
        markers = sorted(d.glob("_SUCCESS_*.json"))
        _require(bool(markers), "run has no _SUCCESS marker")
        sdir, marker_path, expected_result_sha = d, markers[0], None
    else:
        sdir = Path(src["run_dir"])
        if not sdir.is_absolute():
            cands = [Path.cwd() / sdir, d.parents[3] / sdir]
            sdir = next((c for c in cands if c.exists()), cands[0])
        marker_path = sdir / src["marker"]
        expected_result_sha = src.get("result_sha256")
    _require(marker_path.name.startswith("_SUCCESS_") and marker_path.exists(),
             "source run has no _SUCCESS marker")
    marker = json.loads(marker_path.read_bytes())
    results = sorted(sdir.glob("result_*.json"))
    _require(bool(results), "source result missing")
    result_raw = results[0].read_bytes()
    _require(sha256_bytes(result_raw) == marker.get("result_sha256")
             and (expected_result_sha is None or expected_result_sha == marker.get("result_sha256")),
             "source result digest chain broken")
    ledgers = sorted(sdir.glob("call_ledger_*.jsonl"))
    _require(bool(ledgers) and sha256_bytes(ledgers[0].read_bytes()) == marker.get("call_ledger_sha256"),
             "source call ledger digest chain broken")
    rusts = sorted(sdir.glob("rust_report_*.json"))
    _require(bool(rusts) and sha256_bytes(rusts[0].read_bytes()) == marker.get("rust_report_sha256"),
             "source rust report digest chain broken")
    _require((marker.get("rust_report") or {}).get("conformant") is True,
             "source rust report not conformant")
    adapter = _build_adapter(design)
    population = json.loads(result_raw)["population"]
    pop_by_digest = {loci_canonical_digest(adapter.loci_view(p["genes"])): p["genes"]
                     for p in population}

    rrows = _recompute_rows(rows, adapter, terminal, d / "responses", pop_by_digest)
    rec = _summaries(rrows)
    _require(_same(rec["m2"], rep["m2"]), "M2 summary does not match the recomputed evidence")
    og = rep["o_gates"]
    for arm in ("O1", "O2", "O3"):
        got = {k: v for k, v in og[arm].items() if k not in ("calls_used", "distance_strata")}
        _require(_same(rec["o_gates"][arm], got), f"{arm} summary does not match the recomputed evidence")
    n = n_diag
    R = rec["o_gates"]
    exact = (R["O1"]["attempts"] == n and R["O2"]["init_attempts"] == n // 2
             and R["O3"]["init_attempts"] == n // 2
             and R["O2"]["cross_attempts"] == 2 * (R["O2"]["init_valid"] // 2)
             and R["O3"]["mutation_attempts"] == R["O3"]["init_valid"])
    _require(og.get("attempts_exact") is True and exact, "attempts_exact is not true")
    for s, v in rep["m2"].items():
        _require(_finite(v.get("child_diversity_entropy")), f"M2 {s} entropy not finite")

    # ── declared stop rule, T* ──────────────────────────────────
    o1s = og["O2"]["arm_utility"].get("o1_same_budget")
    _require(not _o_unsuitable(og["O2"].get("paired_beat_best_parent"),
                               og["O2"]["arm_utility"].get("half_plus_children"), o1s),
             "O2 stop condition fired (crossover arm unsuitable)")
    _require(not _o_unsuitable(og["O3"].get("paired_beat_parent"),
                               og["O3"]["arm_utility"].get("half_plus_mutants"), o1s),
             "O3 stop condition fired (mutation arm unsuitable)")
    t = calibrated_temperature((rep.get("m3") or {}).get("flip_temperatures"))
    _require(_finite(rep["t_star_e4_rule"]) and abs(t - rep["t_star_e4_rule"]) < 1e-12 and t > 0,
             "T* does not follow the declared rule")
    return rep
