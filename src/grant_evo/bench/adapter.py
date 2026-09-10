"""ProblemAdapter contract for the T-GADE benchmark harness.

A problem plugs into the production evolution core ONLY through this interface.
Runners never re-implement candidate assembly, selection, lineage, or manifests
. The adapter owns problem content (genes),
its operators (LLM-backed or mechanical), its energy normalisation (C5) and its
diversity representation (C6). The core owns everything else.

Contract highlights:
- energy(genes) returns E in [0, 1] (C4/C5) or None = MISSING measurement (C7,
  never zero). The adapter documents its normalisation in `energy_spec`.
- diversity_matrix(genes) returns a float array of shape (num_sections, dim)
  or None = missing; kernel/units are described in `diversity_spec` (C6).
- Operators return new genes or None = FAILURE. The core records the failure
  and leaves a vacancy; it never silently admits a parent copy (C1).
"""

from __future__ import annotations

import random  # noqa: F401  (re-exported convenience for adapters)
from typing import Protocol

import numpy as np

from grant_evo.tgade.engine import Event, Individual, LLMClient  # noqa: F401

class ProblemAdapter(Protocol):
    name: str
    num_sections: int  # M in F = sum E - T * sum_k logdet(L_k + eps I)
    dim: int  # embedding dimensionality D
    energy_spec: str  # how raw objective maps to E in [0,1] (C5)
    diversity_spec: str  # kernel/units of the diversity rows (C6)

    def init_genes(self, rng: random.Random, llm: LLMClient) -> dict | None: ...

    def mutate(
        self, genes: dict, strength: str, rng: random.Random, llm: LLMClient
    ) -> dict | None: ...

    def crossover(
        self, a: dict, b: dict, rng: random.Random, llm: LLMClient
    ) -> dict | None: ...

    def energy(self, genes: dict) -> float | None: ...

    def diversity_matrix(self, genes: dict) -> np.ndarray | None: ...

    def validate(self, genes: dict) -> tuple[bool, str]: ...
