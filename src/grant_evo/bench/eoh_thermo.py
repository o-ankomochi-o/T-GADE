"""Thermodynamic survival rule for the EoH steady-state loop.

Source: the free-energy selection of the thermodynamical genetic algorithm (TDGA,
Mori, Tsukiyama and Kita, PPSN IV, 1996), applied as a removal rule: add the newcomer
to the population (N+1), then remove the ONE member i whose removal minimises the
free energy of the remainder,

    F(P minus i) = sum_{j != i} E_j  -  T * sum_k logdet( G_k(P minus i) + eps I ),

with G_k the population-space Gram matrix of neighbourhood section k (T-GADE extensive
form, free_energy.py).  Two deliberate departures from the memory-population variant, both needed for the
T=0 identity with EoH's ``population_management``:
  * the newcomer is NOT protected (EoH drops an offspring that is the worst);
  * the exclusion policy is a parameter instead of "no identical individuals":
      level : at most one member per objective value, first occurrence kept
              (exactly EoH's objective dedupe)  -> T=0 reproduces EoH bit for bit
      none  : no exclusion; degenerate energy levels and identical codes may coexist
              ("boson" in the sense that any number of members may share a state).
There is no copy mechanism in a removal rule, so "boson" here never means N copies of
the best (that degenerate case belongs to the greedy-addition rule, selection.py).

The returned list is sorted by objective ascending, because EoH's parent_selection
ranks individuals by list index.  The rule is greedy: when more than one member must
go (initial 2N -> N), removals are applied one at a time.
"""
from __future__ import annotations

import numpy as np

from grant_evo.tgade.free_energy import gram_matrix, logdet


def thermo_management(pop, size, *, temperature, rows, energy, exclusion="level",
                      eps=1e-3, trace=None):
    """Survival rule replacing EoH ``population_management(pop, size)``.

    pop         : EoH individual dicts (``objective`` None = failed evaluation, dropped).
    size        : target population size N.
    temperature : T >= 0 of the extensive free energy.
    rows        : callable ind -> (M, D) float array of neighbourhood rows, or None.
                  Only called when T > 0.  A member without rows cannot be placed in
                  the Gram matrix and is excluded first (counted in the trace).
    energy      : callable ind -> E in [0, 1] (the T-GADE arm's clip of the objective).
    exclusion   : "level" (EoH dedupe: one member per objective value), "genotype"
                  (Fermi-type: one member per genotype = description + code) or "none".
    trace       : optional list; one dict per call is appended for audit.
    """
    caps = {"level": 1, "level2": 2, "genotype": 1, "none": None}  # level2: at most two per objective level
    if exclusion == "genotype":
        for ind in pop:  # every individual, including failed evaluations
            if "algorithm" not in ind or "code" not in ind:
                raise ValueError("genotype exclusion needs both description and code on every individual")
    key = (lambda ind: (ind["algorithm"], ind["code"])) if exclusion == "genotype" else (lambda ind: ind["objective"])
    if exclusion not in caps:
        raise ValueError(f"unknown exclusion policy: {exclusion!r}")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    pool = [ind for ind in pop if ind.get("objective") is not None]
    n_in = len(pool)
    dropped_level = 0
    if caps[exclusion] is not None:
        seen, unique = {}, []
        for ind in pool:  # first arrivals keep their state (EoH keeps the first)
            k = key(ind)
            if seen.get(k, 0) < caps[exclusion]:
                seen[k] = seen.get(k, 0) + 1
                unique.append(ind)
        dropped_level = len(pool) - len(unique)
        pool = unique
    dropped_norow = 0
    mats = None
    if temperature > 0 and len(pool) > size:
        placed, mats = [], []
        for ind in pool:
            m = rows(ind)
            if m is None:
                dropped_norow += 1
                continue
            placed.append(ind)
            mats.append(np.asarray(m, dtype=np.float64))
        pool = placed
    removed = []
    newest = pool[-1] if pool else None
    while len(pool) > size:
        e = np.array([float(energy(ind)) for ind in pool])
        total = float(e.sum())
        f = np.empty(len(pool))
        for i in range(len(pool)):
            f[i] = total - e[i]
            if temperature > 0:
                keep = [j for j in range(len(pool)) if j != i]
                ld = 0.0
                for k in range(mats[0].shape[0]):
                    x = np.stack([mats[j][k] for j in keep])
                    ld += logdet(gram_matrix(x, eps=eps))
                f[i] -= temperature * ld
        best = 0
        for i in range(1, len(pool)):
            # ties in F -> remove the larger objective first (the energy clip at RAW_MAX makes

            # (EoH's stable order keeps the first arrival)
            if f[i] < f[best] - 1e-12:
                best = i
            elif abs(f[i] - f[best]) <= 1e-12 and pool[i]["objective"] >= pool[best]["objective"]:
                best = i
        gone = pool.pop(best)
        if mats is not None:
            mats.pop(best)
        removed.append({"objective": gone["objective"], "F": float(f[best]),
                        "is_newcomer": gone is newest})
    pool = sorted(pool, key=lambda ind: ind["objective"])  # stable, like heapq.nsmallest
    if caps[exclusion] == 1:
        # Safety valve: one survivor per state (genotype or objective value).
        states = [key(ind) for ind in pool]
        if len(set(states)) != len(states):
            raise RuntimeError(f"{exclusion} exclusion invariant violated: repeated state among survivors")
    # second occupant: the newcomer survived and shares its objective level with another survivor
    second = (newest is not None and any(x is newest for x in pool)
              and sum(1 for x in pool if x["objective"] == newest["objective"]) >= 2)
    if trace is not None:
        trace.append({"n_in": n_in, "size": size, "T": float(temperature), "exclusion": exclusion,
                      "dropped_level": dropped_level, "dropped_norow": dropped_norow,
                      "removed": removed, "objectives_out": [ind["objective"] for ind in pool],
                      "second_occupant_admitted": bool(second)})
    return pool
