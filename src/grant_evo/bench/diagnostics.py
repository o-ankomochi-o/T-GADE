"""Mandatory behavioural gates M1-M4 and operator gates O1-O3 (invariants are
asserted, behavioural quantities are MEASURED and reported — monotonicity is a
hypothesis, never a pass/fail law).

Everything here is adapter-generic and mock-runnable ($0): the functions call
only the ProblemAdapter surface and the production selector. Real-call usage
just injects real clients. No function here fires network calls of its own.

M1  T=0 behaviour: boson selection at T=0 fills every slot with the argmin
    candidate (invariant, asserted); returns the selected multiset for report.
M2  Perturbation-strength ladder: per strength, child diversity (Vendi entropy
    of adapter diversity rows) and dE dispersion vs parents. Measured only.
M3  Temperature band: per-greedy-step flip temperatures T* = dE / dH between
    the top-2 candidates, plus the random-pool
    dE distribution; reports where the calibrated T sits.
M4  Free-energy bookkeeping of a finished run: per generation Sum E, logdet H,
    F, origin-wise survivor inclusion, clone count, inferior-survivor count.
O1  Random baseline: n inits, energy distribution.
O2  Crossover: n/2 inits + n/2 children (random disjoint pairs); paired
    beat-best-parent rate AND same-budget arm utility, reported separately
; near/far parent-distance strata disclosed
    as mechanism analysis only.
O3  Mutation: n/2 inits + one mutate each; same two estimands.
"""

from __future__ import annotations

import math
import random

import numpy as np

from grant_evo.tgade.free_energy import gram_matrix as _gram_matrix, logdet as _logdet
from grant_evo.tgade.selection import thermodynamical_select


def gram_logdet(rows, *, eps: float) -> float:
    """log det(X Xᵀ + εI) of an (N, D) row matrix; scripts reach the kernel through bench."""
    return _logdet(_gram_matrix(np.asarray(rows, dtype=np.float64), eps=eps))


def _entropy_of_rows(rows: list[np.ndarray]) -> float:
    """Vendi-style kernel eigenvalue entropy of unit-normalised rows (nats)."""
    if len(rows) < 2:
        return 0.0
    m = np.stack([r.reshape(-1) for r in rows]).astype(float)
    nrm = np.linalg.norm(m, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    m = m / nrm
    g = (m @ m.T) / len(rows)
    ev = np.clip(np.linalg.eigvalsh(g), 0.0, None)
    s = ev.sum()
    if s <= 1e-12:
        return 0.0
    p = ev / s
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def _measure(adapter, genes, llm=None):
    """energy + diversity row for one genotype; None if missing (C7)."""
    e = adapter.energy(genes)
    if e is None:
        return None
    d = adapter.diversity_matrix(genes)
    if d is None:
        return None
    return float(e), np.asarray(d, dtype=float)


def attempt_pool(adapter, llm, rng: random.Random, attempts: int, sink=None,
                 phase: str = "init"):
    """Exactly `attempts` init attempts, NO retries (fixed-attempt accounting):
    returns (valid rows, attempts, failures). Uses the engine's shared
    viability check. sink(row) receives one evidence row per
    attempt."""
    from grant_evo.tgade.engine import viability  # noqa: PLC0415
    pool, fails = [], 0
    for k in range(attempts):
        g = adapter.init_genes(rng, llm)
        bind = _bind(llm)
        ok, kind, why, e, m = (viability(adapter, g) if g is not None
                               else (False, "operator_failure", "init returned None", None, None))
        if not ok:
            fails += 1
            if sink:
                sink({"phase": phase, "attempt": k, "ok": False,
                      "reason": f"{kind}: {why}", **bind})
            continue
        pool.append({"genes": g, "energy": e, "div": m, "attempt": k})
        if sink:
            sink({"phase": phase, "attempt": k, "ok": True, "digest": _digest(adapter, g),
                  "energy": e, "div": np.asarray(m, dtype=float).reshape(-1).tolist(),
                  **bind})
    return pool, attempts, fails


def build_pool(adapter, llm, rng: random.Random, n: int, max_tries: int = 4):
    """n evaluated random individuals (init operator). Fail-closed on shortfall."""
    pool = []
    tries = 0
    while len(pool) < n and tries < max_tries * n:
        tries += 1
        g = adapter.init_genes(rng, llm)
        if g is None:
            continue
        ok, _ = adapter.validate(g)
        if not ok:
            continue
        m = _measure(adapter, g)
        if m is None:
            continue
        pool.append({"genes": g, "energy": m[0], "div": m[1]})
    if len(pool) < n:
        raise RuntimeError(f"build_pool: only {len(pool)}/{n} valid after {tries} tries")
    return pool


# ── M1 ────────────────────────────────────────────────────────────────
def m1_t0_check(adapter, pool) -> dict:
    ids = [f"p{i}" for i in range(len(pool))]
    res = thermodynamical_select(
        candidate_ids=ids,
        energies={i: p["energy"] for i, p in zip(ids, pool)},
        embeddings={i: p["div"] for i, p in zip(ids, pool)},
        target_size=min(4, len(pool)),
        num_sections=adapter.num_sections,
        dim=adapter.dim,
        temperature=0.0,
        occupancy="boson",
    )
    argmin = ids[int(np.argmin([p["energy"] for p in pool]))]
    assert res.selected_ids == [argmin] * len(res.selected_ids), (
        f"M1 INVARIANT VIOLATION: T=0 boson selection {res.selected_ids} != all-{argmin}")
    return {"selected": res.selected_ids, "argmin": argmin, "invariant": "pass"}


# ── M2 ────────────────────────────────────────────────────────────────
def _digest(adapter, genes) -> "str | None":
    from grant_evo.tgade.engine import loci_canonical_digest  # noqa: PLC0415
    view = getattr(adapter, "loci_view", None)
    return loci_canonical_digest(view(genes)) if view else None


def m2_strength_ladder(adapter, llm, rng: random.Random, pool,
                       strengths=("weak", "mid", "strong"), children_per_parent=2,
                       sink=None) -> dict:
    """sink(row): optional per-attempt evidence row ( the
    E3 gate recomputes every summary from these rows)."""
    out = {}
    acc = {s: {"rows": [], "des": [], "fails": 0} for s in strengths}
    # COUNTERBALANCED order: strengths are interleaved
    # per (parent, child) with a rotating start so no strength is confounded
    # with call index / time / provider-seed drift.
    k = len(strengths)
    for i, p in enumerate(pool):
        for j in range(children_per_parent):
            for t in range(k):
                s = strengths[(i + j + t) % k]
                a = acc[s]
                row = {"phase": "m2", "parent_index": i, "child_index": j, "strength": s,
                       "parent_digest": _digest(adapter, p["genes"]),
                       "parent_energy": p["energy"], "ok": False}
                g = adapter.mutate(dict(p["genes"]), s, rng, llm)
                row.update(_bind(llm))
                m = _measure(adapter, g) if g is not None else None
                if m is None:
                    a["fails"] += 1
                    if sink:
                        sink(row)
                    continue
                a["rows"].append(m[1])
                a["des"].append(m[0] - p["energy"])
                if sink:
                    sink({**row, "ok": True, "child_digest": _digest(adapter, g),
                          "energy": m[0], "dE": m[0] - p["energy"],
                          "div": np.asarray(m[1], dtype=float).reshape(-1).tolist()})
    for s in strengths:
        rows, des, fails = acc[s]["rows"], acc[s]["des"], acc[s]["fails"]
        out[s] = {
            "n_children": len(des),
            "failures": fails,
            "child_diversity_entropy": round(_entropy_of_rows(rows), 4),
            "dE_mean": round(float(np.mean(des)), 5) if des else None,
            "dE_std": round(float(np.std(des)), 5) if des else None,
            "dE_p_improve": round(float(np.mean([d < 0 for d in des])), 3) if des else None,
        }
    return out  # measured; monotonicity is analysed, never asserted


# ── M3 ────────────────────────────────────────────────────────────────
def m3_temperature_band(adapter, pool, temperature: float) -> dict:
    """Greedy-path flip temperatures: at each step, T* between the best and
    second-best candidate by dF; the calibrated T is located in that set."""
    from grant_evo.tgade.free_energy import GramFreeEnergy
    energies = [p["energy"] for p in pool]
    state = GramFreeEnergy(num_sections=adapter.num_sections, dim=adapter.dim,
                           eps=1e-3, temperature=temperature)
    chosen = int(np.argmin(energies))
    state.add("m0", pool[chosen]["div"])
    flips = []
    for step in range(1, min(4, len(pool))):
        stats = []
        for j, p in enumerate(pool):
            dh = state.delta_h(p["div"])
            stats.append((energies[j], dh))
        dfs = [e - temperature * dh for e, dh in stats]
        order = np.argsort(dfs)
        a, b = int(order[0]), int(order[1])
        de = stats[b][0] - stats[a][0]
        dh = stats[b][1] - stats[a][1]
        if abs(dh) > 1e-12:
            flips.append(de / dh)
        state.add(f"m{step}", pool[a]["div"])  # boson: repeats need fresh instance ids
    des = np.abs(np.diff(np.sort(energies)))
    des = des[des > 0]
    return {
        "flip_temperatures": [round(f, 5) for f in flips],
        "calibrated_T": temperature,
        "T_below_all_flips": bool(flips) and temperature < min(flips),
        "T_above_all_flips": bool(flips) and temperature > max(flips),
        "pool_dE_min_nonzero": round(float(des.min()), 6) if des.size else None,
        "pool_dE_median": round(float(np.median(des)), 6) if des.size else None,
    }


# ── M4 ────────────────────────────────────────────────────────────────
def m4_run_report(result) -> list[dict]:
    """Origin-wise survivor accounting + F path from a finished RunResult."""
    by_id = {i.id: i for i in result.lineage}
    out = []
    for log in result.generation_log:
        origins = {"elite_carryover": 0, "parent_variant": 0, "cross": 0}
        inferior = 0
        cand_e = {}
        for m in log["instance_mapping"]:
            src = by_id.get(m["selected_cid"])
            if src is None:
                continue
            origins[src.op] = origins.get(src.op, 0) + 1
            cand_e[m["selected_cid"]] = src.energy
        best_e = log.get("best_energy")
        for cid, e in cand_e.items():
            if e is not None and best_e is not None and e > best_e:
                inferior += 1
        out.append({
            "gen": log["gen"],
            "origin_inclusion": origins,
            "clones": log.get("clones"),
            "distinct_survivor_sources": log.get("distinct_survivor_sources"),
            "final_free_energy": log.get("final_free_energy"),
            "final_logdet": log.get("final_logdet"),
            "survivors_worse_than_best": inferior,
        })
    return out


# ── O gates ───────────────────────────────────────────────────────────
def _summary(es: list[float]) -> dict:
    """n/best/median plus mean and SE so that a mean + 2*SE comparison
    is computable from the report."""
    a = np.asarray(es, dtype=float)
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None
    return {"n": len(es), "best": round(min(es), 5),
            "median": round(float(np.median(a)), 5),
            "mean": round(float(a.mean()), 5),
            "se": round(se, 6) if se is not None else None}


def calibrated_temperature(flip_temperatures, fallback: float = 0.5) -> float:
    """Calibrated temperature: geometric mean of the finite POSITIVE flip
    temperatures; pre-declared fallback when fewer than 3 qualify."""
    vals = [float(t) for t in (flip_temperatures or [])
            if np.isfinite(t) and t > 0]
    if len(vals) < 3:
        return fallback
    return float(np.exp(np.mean(np.log(vals))))


class _CountingLLM:
    """Counts every underlying call so O-gate budgets are AUDITABLE
."""

    def __init__(self, llm):
        self._llm = llm
        self.calls = 0

    def __call__(self, prompt, **opts):
        self.calls += 1
        return self._llm(prompt, **opts) if opts else self._llm(prompt)

    def __getattr__(self, name):  # last_seq / last_response_sha256 passthrough
        return getattr(self._llm, name)


def _bind(llm) -> dict:
    """Ledger binding of the call just made: the
    DiagRecorder exposes the seq and response digest of its last call."""
    return {"seq": getattr(llm, "last_seq", None),
            "response_sha256": getattr(llm, "last_response_sha256", None)}


def o_gates(adapter, llm, rng: random.Random, n: int = 8, strength: str = "mid",
            sink=None) -> dict:
    if n % 4 != 0:
        raise ValueError("o_gates requires n % 4 == 0 (O2 needs n/2 children "
                         "from n/4 disjoint pairs in both orders;)")
    if n % 2:
        raise ValueError("n must be even")
    c = _CountingLLM(llm)
    llm = c
    # FIXED-ATTEMPT accounting (the former
    # retry-until-N pools made "same budget" unattainable with real models):
    # every O arm spends exactly its nominal attempt quota, failures are
    # counted and reported, utilities are computed over VALID candidates.
    o1, o1_att, o1_fail = attempt_pool(adapter, llm, rng, n, sink=sink, phase="o1")
    e1 = [p["energy"] for p in o1]
    o1_calls = c.calls

    half, half_att, half_fail = attempt_pool(adapter, llm, rng, n // 2, sink=sink,
                                             phase="o2_init")
    order = list(range(len(half)))
    rng.shuffle(order)
    kids, paired, dists, fails, cross_att = [], [], [], 0, 0
    for i in range(0, len(order) - 1, 2):
        a, b = half[order[i]], half[order[i + 1]]
        for pa, pb in ((a, b), (b, a)):
            cross_att += 1
            row = {"phase": "o2_cross",
                   "pair": [order[i], order[i + 1]],  # indices into VALID inits
                   "parent_attempts": [pa["attempt"], pb["attempt"]],  # o2_init attempt ids
                   "parent_energies": [pa["energy"], pb["energy"]], "ok": False}
            g = adapter.crossover(dict(pa["genes"]), dict(pb["genes"]), rng, llm)
            row.update(_bind(llm))
            m = _measure(adapter, g) if g is not None else None
            if m is None:
                fails += 1
                if sink:
                    sink(row)
                continue
            kids.append(m[0])
            beat = bool(m[0] < min(pa["energy"], pb["energy"]) - 1e-12)
            paired.append(beat)
            va, vb = pa["div"].reshape(-1), pb["div"].reshape(-1)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            dist = 1.0 - float(va @ vb) / (na * nb + 1e-12)
            dists.append(dist)
            if sink:
                sink({**row, "ok": True, "digest": _digest(adapter, g), "energy": m[0],
                      "div": np.asarray(m[1], dtype=float).reshape(-1).tolist(),
                      "beat_best_parent": beat, "dist": dist})
    strata = {}
    if dists:
        med = float(np.median(dists))
        near = [k for k, d in zip(kids, dists) if d <= med]
        far = [k for k, d in zip(kids, dists) if d > med]
        strata = {"near_median_best": round(min(near), 5) if near else None,
                  "far_median_best": round(min(far), 5) if far else None,
                  "note": "mechanism analysis only (multiplicity disclosed)"}
    o2 = {
        "arm_utility": {"o1_same_budget": _summary(e1),
                        "half_plus_children": _summary([p["energy"] for p in half] + kids)
                        if kids else None},
        "paired_beat_best_parent": round(float(np.mean(paired)), 3) if paired else None,
        "n_children": len(kids), "failures": fails, "distance_strata": strata,
        "init_attempts": half_att, "init_failures": half_fail,
        "init_valid": len(half), "cross_attempts": cross_att,
        "calls_used": c.calls - o1_calls,
    }
    o2_end = c.calls

    half3, half3_att, half3_fail = attempt_pool(adapter, llm, rng, n // 2, sink=sink,
                                                phase="o3_init")
    kids3, paired3, fails3, mut_att = [], [], 0, 0
    for pi, p in enumerate(half3):
        mut_att += 1
        row = {"phase": "o3_mut", "parent_index": pi, "parent_attempt": p["attempt"],
               "parent_energy": p["energy"], "strength": strength, "ok": False}
        g = adapter.mutate(dict(p["genes"]), strength, rng, llm)
        row.update(_bind(llm))
        m = _measure(adapter, g) if g is not None else None
        if m is None:
            fails3 += 1
            if sink:
                sink(row)
            continue
        kids3.append(m[0])
        beat = bool(m[0] < p["energy"] - 1e-12)
        paired3.append(beat)
        if sink:
            sink({**row, "ok": True, "digest": _digest(adapter, g), "energy": m[0],
                  "div": np.asarray(m[1], dtype=float).reshape(-1).tolist(),
                  "beat_parent": beat})
    o3 = {
        "arm_utility": {"o1_same_budget": _summary(e1) if e1 else None,
                        "half_plus_mutants": _summary([p["energy"] for p in half3] + kids3)
                        if kids3 else None},
        "paired_beat_parent": round(float(np.mean(paired3)), 3) if paired3 else None,
        "n_children": len(kids3), "failures": fails3,
        "init_attempts": half3_att, "init_failures": half3_fail,
        "init_valid": len(half3), "mutation_attempts": mut_att,
        "calls_used": c.calls - o2_end,
    }
    if e1:
        o2["arm_utility"]["o1_same_budget"] = _summary(e1)
    # attempts_exact: every arm spent exactly its nominal attempt quota, no
    # retries: O1 n inits; O2 n/2 inits + one crossover attempt per ordered
    # disjoint pair of VALID inits; O3 n/2 inits + one mutation attempt per
    # valid init. Failures never break exactness - they are reported.
    attempts_exact = (o1_att == n and half_att == n // 2 and half3_att == n // 2
                      and cross_att == 2 * (len(half) // 2)
                      and mut_att == len(half3))
    return {"O1": {**(_summary(e1) if e1 else {"n": 0}), "attempts": o1_att,
                   "failures": o1_fail, "calls_used": o1_calls},
            "O2": o2, "O3": o3,
            "attempts_exact": attempts_exact,
            "note": "descriptive; arm-utility vs paired-gain are separate "
                    "estimands; fixed attempt quotas, failures reported; "
                    "same-budget comparison valid ONLY if attempts_exact"}
