"""Step 8 — thermodynamical greedy selection (T-GADE paper §3.4 Step 8).

Build P_{t+1} (size N) from P' (size > N) by:

1. Apply feasibility filter: candidates with constraint_violations > 0 are
   excluded from the pool (T-GADE paper §3.2 end + Algorithm 1 Step 8
   comment). If all candidates are infeasible, fall back to keeping the
   minimum-violation candidates (logged warning).
2. Move the lowest-energy feasible candidate to P_{t+1}.
3. While |P_{t+1}| < N, evaluate Δ F(y) for each feasible candidate using
   the GramFreeEnergy state (Schur-complement update, O(|P_{t+1}|^2) per
   candidate). Move the y* that minimises Δ F.
4. Return the constructed P_{t+1} along with a per-step trace for
   debugging / observability.

Occupancy (author ruling 2026-09-02, TDGA 1998/2021 lineage):
the DEFAULT is ``occupancy='boson'`` — a candidate stays in the pool
after being selected, so duplicate survivors (clones) are legal and the
population can condense onto few genotypes at low T. ``'fermion'``
(selection without replacement, the pre-2026-09 engine behaviour) is an
explicit special option only. Clones enter the Gram state under
occurrence-suffixed instance ids (``cid#1``, ``cid#2``, …); a duplicate
embedding row contributes ΔH ≈ log ε, i.e. the determinantal functional
penalises repetition smoothly but never forbids it.

The engine minimises the total objective Σ_x E(x) - T Σ_k log det L_k
(``free_energy_mode='extensive'``, the default). This is |P|·F_T(P) for
the paper's free energy F_T(P) = ⟨E⟩_P - T·H(P) with
H(P) = (1/|P|) Σ_k log det L_k; every selection step compares sets of
equal size, so the minimiser is the same. The ``'intensive'`` mode
(⟨E⟩ - T Σ_k log det L_k, no 1/|P| on the diversity term) is a legacy
ablation hook and is not the paper's F_T: the two modes do NOT in general
produce the same greedy trajectory at fixed N.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from grant_evo.tgade.free_energy import GramFreeEnergy


@dataclass
class SelectionStep:
    """One greedy step's diagnostic record."""

    step: int
    selected_id: str
    energy: float
    delta_f: float
    pool_remaining: int


@dataclass
class SelectionResult:
    selected_ids: list[str]
    trace: list[SelectionStep]
    final_free_energy: float
    final_logdet: float


def _prepare_candidates(
    *,
    candidate_ids: list[str],
    energies: dict[str, float],
    embeddings: dict[str, np.ndarray],
    target_size: int,
    num_sections: int,
    dim: int,
    constraint_violations: dict[str, int] | None,
) -> list[str]:
    """Validate the shared selector contract and return unique ids in caller order."""
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if num_sections <= 0:
        raise ValueError("num_sections must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    pool = list(dict.fromkeys(candidate_ids))
    if len(pool) < target_size:
        raise ValueError(f"unique pool size {len(pool)} < target_size {target_size}")

    missing_e = [candidate_id for candidate_id in pool if candidate_id not in energies]
    if missing_e:
        raise ValueError(f"missing energies for: {missing_e[:5]}...")
    missing_x = [candidate_id for candidate_id in pool if candidate_id not in embeddings]
    if missing_x:
        raise ValueError(f"missing embeddings for: {missing_x[:5]}...")

    for candidate_id in pool:
        energy = float(energies[candidate_id])
        if not math.isfinite(energy) or not 0.0 <= energy <= 1.0:
            raise ValueError(
                f"energy for {candidate_id!r} must be finite and in [0, 1], got {energy!r}"
            )
        vector = embeddings[candidate_id]
        if vector.shape != (num_sections, dim):
            raise ValueError(
                f"embedding for {candidate_id!r} must have shape "
                f"({num_sections}, {dim}), got {vector.shape}"
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"embedding for {candidate_id!r} contains non-finite values")

    if constraint_violations is not None:
        for candidate_id in pool:
            count = constraint_violations.get(candidate_id, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    f"constraint violations for {candidate_id!r} must be a non-negative int"
                )
    return pool


def _apply_feasibility_filter(
    *,
    pool: list[str],
    energies: dict[str, float],
    target_size: int,
    constraint_violations: dict[str, int] | None,
    occupancy: str = "boson",
    genotype_of: "dict[str, str] | None" = None,
) -> list[str]:
    """Apply the common feasible-first policy used by every selector.

    Under Fermi-type occupancy the shortfall is counted in distinct genotypes
    (one individual per genotype), not in candidate ids: the minimum-violation
    candidates are admitted until ``target_size`` distinct genotypes are present,
    and an infeasible twin of a feasible genotype is never admitted.

    Under boson occupancy ANY non-empty feasible set can fill the target by
    duplication, so infeasible candidates are admitted only when no feasible
    candidate exists at all ( the fermionic shortfall
    fallback must not leak infeasible sources into a boson selection)."""
    if constraint_violations is None:
        return pool
    feasible = [candidate_id for candidate_id in pool if constraint_violations.get(candidate_id, 0) == 0]
    if occupancy == "boson" and feasible:
        return feasible
    infeasible = sorted(
        (candidate_id for candidate_id in pool if constraint_violations.get(candidate_id, 0) > 0),
        key=lambda candidate_id: (
            constraint_violations.get(candidate_id, 0),
            energies[candidate_id],
        ),
    )
    if occupancy == "fermion" and genotype_of:
        chosen, seen = list(feasible), {genotype_of[c] for c in feasible}
        for candidate_id in infeasible:
            if len(seen) >= target_size:
                break
            if genotype_of[candidate_id] not in seen:
                chosen.append(candidate_id)
                seen.add(genotype_of[candidate_id])
        return chosen
    if len(feasible) >= target_size:
        return feasible
    return feasible + infeasible[: target_size - len(feasible)]


def _check_genotype_map(pool: list, genotype_of: "dict[str, str] | None") -> None:
    """Fermi-type contract: a genotype map covering the whole pool is required."""
    if not genotype_of:
        raise ValueError("fermion occupancy requires genotype_of (one individual per genotype)")
    missing = [c for c in pool if c not in genotype_of]
    if missing:
        raise ValueError(f"fermion occupancy: genotype_of lacks {missing[:5]}")


def assert_fermi_invariant(selected: list, genotype_of: "dict[str, str]") -> None:
    """Safety valve: under Fermi-type occupancy every selected individual must be a
    distinct candidate AND a distinct genotype. Raises RuntimeError otherwise."""
    if len(set(selected)) != len(selected):
        raise RuntimeError(f"fermion invariant violated: repeated candidate id in {selected}")
    genos = [genotype_of[c] for c in selected]
    if len(set(genos)) != len(genos):
        dup = [c for c in selected if genos.count(genotype_of[c]) > 1]
        raise RuntimeError(f"fermion invariant violated: repeated genotype among {dup}")


def _exclude_genotype(pool: list, chosen: str, genotype_of: "dict[str, str] | None") -> list:
    """Fermi-type occupancy: drop the chosen id and every candidate with the same genotype."""
    g = genotype_of.get(chosen) if genotype_of else None
    return [c for c in pool if c != chosen and (g is None or genotype_of.get(c) != g)]


def thermodynamical_select(
    *,
    candidate_ids: list[str],
    energies: dict[str, float],
    embeddings: dict[str, np.ndarray],
    target_size: int,
    num_sections: int,
    dim: int,
    eps: float = 1e-3,
    temperature: float = 0.5,
    constraint_violations: dict[str, int] | None = None,
    free_energy_mode: str = "extensive",
    occupancy: str = "boson",
    genotype_of: dict[str, str] | None = None,
) -> SelectionResult:
    """Greedy selection of ``target_size`` candidates minimising free energy.

    Parameters
    ----------
    candidate_ids : individuals to choose from.
    energies : id → E(x) ∈ [0, 1].
    embeddings : id → (M, D) float array.
    target_size : final population size N.
    num_sections : M.
    dim : embedding dimensionality D.
    eps : Gram-matrix regulariser.
    temperature : T of the paper's free energy F_T = <E> - T·H with H = (1/|P|) Σ_k log det L_k;
        the implementation minimises the total objective |P|·F_T = Σ E - T · Σ_k log det L_k,
        which has the same minimiser because compared sets have equal size.
    genotype_of : id → canonical genotype key. Under Fermi-type occupancy
        every candidate whose genotype equals a selected one is removed from
        the pool (one individual per genotype). Required under Fermi-type occupancy
        (ValueError otherwise); unused under Bose-type occupancy.
    constraint_violations : id → ν(x), the per-individual violation count.
        Optional. When provided, candidates with ν > 0 are excluded from
        the pool first (T-GADE paper §3.2 + Algorithm 1 Step 8). If after
        filtering the pool is smaller than target_size, the filter falls
        back to admitting the minimum-violation candidates needed to fill
        the target. When omitted, behaviour is unchanged (no filtering).
    free_energy_mode : "extensive" (default; the total objective
        |P|·F_T = Σ E - T Σ_k log det L_k, same minimiser as the paper's
        mean-form F_T) or "intensive" (legacy ablation, not the paper's F_T).
        Controls both the running increment in the greedy loop and the
        reported final value.
    occupancy : "boson" (DEFAULT, author ruling 2026-09-02): selection
        WITH replacement — a picked candidate stays in the pool, duplicate
        survivors are legal and appear repeated in ``selected_ids``.
        "fermion": selection without replacement (legacy engine
        behaviour), unique pool of size ≥ target_size required.

    Returns
    -------
    SelectionResult with selected_ids in selection order and a per-step trace.
    Under boson occupancy ``selected_ids`` is a multiset (repeats allowed).

    Raises
    ------
    ValueError if pool too small (fermion), empty (boson), or shape mismatches.
    """
    if free_energy_mode not in {"extensive", "intensive"}:
        raise ValueError(f"unknown free_energy mode: {free_energy_mode!r}")
    if occupancy not in {"boson", "fermion"}:
        raise ValueError(f"unknown occupancy: {occupancy!r}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    pool = _prepare_candidates(
        candidate_ids=candidate_ids,
        energies=energies,
        embeddings=embeddings,
        # boson: duplicates can fill the target, so only a non-empty unique
        # pool is required; fermion keeps the strict ≥ target_size contract.
        target_size=1 if occupancy == "boson" else target_size,
        num_sections=num_sections,
        dim=dim,
        constraint_violations=constraint_violations,
    )

    # Feasibility filter (paper §3.2 end). When constraint_violations is
    # None, behaviour is unchanged. When supplied, ν > 0 candidates are
    # excluded; if too few feasible remain, we admit the lowest-ν infeasible
    # candidates in order to reach target_size (Fermi-type: counted in distinct
    # genotypes; fallback policy: degrade gracefully rather than abort the generation).
    if occupancy == "fermion":
        _check_genotype_map(pool, genotype_of)
    pool = _apply_feasibility_filter(
        pool=pool,
        energies=energies,
        target_size=target_size,
        constraint_violations=constraint_violations,
        occupancy=occupancy,
        genotype_of=genotype_of,
    )
    state = GramFreeEnergy(
        num_sections=num_sections,
        dim=dim,
        eps=eps,
        temperature=temperature,
    )
    occurrence: dict[str, int] = {}

    def _instance_id(cid: str) -> str:
        """Occurrence-suffixed id so boson clones coexist in the Gram state.
        Uses the unit-separator control char, which cannot collide with any
        caller-supplied candidate id."""
        n = occurrence.get(cid, 0)
        occurrence[cid] = n + 1
        return cid if n == 0 else f"{cid}\x1f{n}"

    # Step 1: lowest-energy seed — EXACTLY as ALGORITHM.md §0 Step 8.1 writes
    # it ( ruling: the spec is canonical; the Rust oracle, which
    # seeds by empty-state ΔF, must be aligned to this rule, not vice versa).
    # trace[0] records the raw seed energy with an explicit bootstrap marker;
    # tie-break = first in caller order (contract C2-tie, pending Rust echo).
    if occupancy == "fermion":
        distinct = len({genotype_of[c] for c in pool})
        if distinct < target_size:
            raise ValueError(
                f"fermion occupancy needs {target_size} distinct genotypes, pool has {distinct}")
    seed = min(pool, key=lambda c: energies[c])
    state.add(_instance_id(seed), embeddings[seed])
    if occupancy == "fermion":
        pool = _exclude_genotype(pool, seed, genotype_of)
    selected = [seed]
    trace: list[SelectionStep] = [
        SelectionStep(
            step=0,
            selected_id=seed,
            energy=energies[seed],
            delta_f=energies[seed],  # bootstrap: raw E, no prior Gram state
            pool_remaining=len(pool),
        )
    ]

    # Step 2: greedy ΔF minimisation.
    while len(selected) < target_size:
        # current_mean_energy is only needed by the intensive ΔF formula.
        # In extensive mode (default) the ΔF expression is just
        # E(y) - T · ΔH(y), and current_mean is ignored downstream.
        if free_energy_mode == "intensive":
            current_mean = sum(energies[s] for s in selected) / float(len(selected))
        else:
            current_mean = None
        best_id: str | None = None
        best_df = float("inf")
        for cid in pool:
            df = state.delta_f_for_greedy(
                candidate_energy=energies[cid],
                candidate_embeddings=embeddings[cid],
                current_mean_energy=current_mean,
                mode=free_energy_mode,
            )
            if df < best_df:
                best_df = df
                best_id = cid
        if best_id is None:
            raise ValueError("fermion occupancy: candidate pool exhausted before target_size")
        state.add(_instance_id(best_id), embeddings[best_id])
        if occupancy == "fermion":
            pool = _exclude_genotype(pool, best_id, genotype_of)
        selected.append(best_id)
        trace.append(
            SelectionStep(
                step=len(selected) - 1,
                selected_id=best_id,
                energy=energies[best_id],
                delta_f=best_df,
                pool_remaining=len(pool),
            )
        )

    if occupancy == "fermion":
        assert_fermi_invariant(selected, genotype_of)

    final_sum = sum(energies[s] for s in selected)
    if free_energy_mode == "intensive":
        final_free_energy = state.free_energy(
            mean_energy=final_sum / float(len(selected)),
            mode="intensive",
        )
    else:
        final_free_energy = state.free_energy(
            sum_energy=final_sum, mode="extensive"
        )
    return SelectionResult(
        selected_ids=selected,
        trace=trace,
        final_free_energy=final_free_energy,
        final_logdet=state.total_logdet(),
    )


def random_select(
    *,
    candidate_ids: list[str],
    energies: dict[str, float],
    embeddings: dict[str, np.ndarray],
    target_size: int,
    num_sections: int,
    dim: int,
    eps: float = 1e-3,
    temperature: float = 0.5,
    constraint_violations: dict[str, int] | None = None,
    rng: random.Random | None = None,
    occupancy: str = "boson",
    genotype_of: dict[str, str] | None = None,
) -> SelectionResult:
    """Random-search baseline: evaluate candidates, then ignore scores.

    This keeps the same evaluated candidate pool as T-GADE Step 8, including
    the feasibility filter, but replaces thermodynamical greedy selection by
    uniform random sampling. Occupancy mirrors ``thermodynamical_select``:
    "boson" (default) samples WITH replacement, "fermion" without.
    """
    if occupancy not in {"boson", "fermion"}:
        raise ValueError(f"unknown occupancy: {occupancy!r}")
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    candidate_ids = _prepare_candidates(
        candidate_ids=candidate_ids,
        energies=energies,
        embeddings=embeddings,
        target_size=1 if occupancy == "boson" else target_size,
        num_sections=num_sections,
        dim=dim,
        constraint_violations=constraint_violations,
    )
    if occupancy == "fermion":
        _check_genotype_map(candidate_ids, genotype_of)
    candidate_ids = _apply_feasibility_filter(
        pool=candidate_ids,
        energies=energies,
        target_size=target_size,
        constraint_violations=constraint_violations,
        occupancy=occupancy,
        genotype_of=genotype_of,
    )

    rng = rng or random.Random()
    if occupancy == "boson":
        selected = [candidate_ids[rng.randrange(len(candidate_ids))] for _ in range(target_size)]
    else:
        selected = rng.sample(candidate_ids, target_size)
        if len({genotype_of[c] for c in selected}) < target_size:
            selected, pool = [], list(candidate_ids)
        while len(selected) < target_size:
            if not pool:
                raise ValueError("fermion occupancy: candidate pool exhausted before target_size")
            pick = pool[rng.randrange(len(pool))]
            selected.append(pick)
            pool = _exclude_genotype(pool, pick, genotype_of)
        assert_fermi_invariant(selected, genotype_of)
    state = GramFreeEnergy(
        num_sections=num_sections,
        dim=dim,
        eps=eps,
        temperature=temperature,
    )
    trace: list[SelectionStep] = []
    occurrence: dict[str, int] = {}
    for step, cid in enumerate(selected):
        if step == 0:
            delta_f = energies[cid]
        else:
            delta_f = state.delta_f_for_greedy(
                candidate_energy=energies[cid],
                candidate_embeddings=embeddings[cid],
                mode="extensive",
            )
        n_occ = occurrence.get(cid, 0)
        occurrence[cid] = n_occ + 1
        state.add(cid if n_occ == 0 else f"{cid}\x1f{n_occ}", embeddings[cid])
        trace.append(
            SelectionStep(
                step=step,
                selected_id=cid,
                energy=energies[cid],
                delta_f=delta_f,
                pool_remaining=max(0, len(candidate_ids) - step - 1),
            )
        )

    final_sum = sum(energies[s] for s in selected)
    return SelectionResult(
        selected_ids=selected,
        trace=trace,
        final_free_energy=state.free_energy(sum_energy=final_sum, mode="extensive"),
        final_logdet=state.total_logdet(),
    )
