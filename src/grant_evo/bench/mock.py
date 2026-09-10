"""Scriptable mock LLM + deterministic mock problem for structural tests (G3).

The mock exists to exercise invariants and failure paths, not performance
. ScriptableLLM injects successes, failures, and malformed output
in a prescribed order; MockProblem is a deterministic adapter whose energy and
diversity depend only on the genes, so golden traces are reproducible.
"""

from __future__ import annotations

import random

import numpy as np

from grant_evo.bench.adapter import LLMClient


class ScriptableLLM:
    """LLM stub. `script` is a list of behaviours consumed per call:
    "ok" -> returns a deterministic completion, "fail" -> None,
    "garbage" -> non-parseable text. When the script is exhausted, "ok"."""

    def __init__(self, script: list[str] | None = None):
        self.script = list(script or [])
        self.calls: list[str] = []

    def __call__(self, prompt: str, **opts) -> str | None:
        self.calls.append(prompt)
        behaviour = self.script.pop(0) if self.script else "ok"
        if behaviour == "fail":
            return None
        if behaviour == "garbage":
            return "%%% not parseable %%%"
        return f"OK:{len(self.calls)}"


class MockProblem:
    """Deterministic toy adapter. genes = {"x": int}.

    energy  = (x mod 97) / 96  in [0,1]  (deterministic, quantised on purpose)
    diversity_matrix = (1, 8) one-hot of x mod 8
    mutate  = x + step(strength); fails when the LLM call fails
    cross   = midpoint; fails when the LLM call fails
    An x with genes["poison"] returns None energy (missing-measurement path).
    """

    name = "mock"
    num_sections = 1
    dim = 8
    parse_predicate = "mock_v1"  # oracle replay: unparseable iff "not parseable"
    energy_spec = "E = (x mod 97)/96; already in [0,1]"
    diversity_spec = "one-hot of x mod 8, unit rows, nats"

    STEP = {"weak": 1, "mid": 3, "strong": 9}

    # Genotypes are PURE FUNCTIONS of (response text, parent genes) so the
    # E3 gate can reconstruct them from the retained raw responses
    #: init x = h(text) mod 1000; mutate x = parent x +
    # (h(text) mod (2*step+1)) - step; crossover = midpoint (text only
    # decides success). "not parseable" = parse failure, None = transport.
    @staticmethod
    def _h(text: str) -> int:
        import hashlib  # noqa: PLC0415
        return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)

    def reconstruct(self, role: str, text: str, parents: list, strength: str = "mid"):
        """Pure parser for the gate: genes or None (parse failure)."""
        if text is None or "not parseable" in str(text):
            return None
        if role == "init":
            return {"x": self._h(text) % 1000}
        if role == "mutate":
            step = self.STEP.get(strength, 3)
            return {**parents[0], "x": parents[0]["x"] + (self._h(text) % (2 * step + 1)) - step}
        if role == "cross":
            return {"x": (parents[0]["x"] + parents[1]["x"]) // 2}
        if role == "integrity":
            return dict(parents[0])
        raise ValueError(role)

    def init_genes(self, rng: random.Random, llm: LLMClient) -> dict | None:
        return self.reconstruct("init", llm("init"), [])

    def mutate(self, genes: dict, strength: str, rng: random.Random,
               llm: LLMClient, template: str | None = None) -> dict | None:
        return self.reconstruct("mutate", llm("mutate"), [genes], strength)

    def crossover(self, a: dict, b: dict, rng: random.Random,
                  llm: LLMClient, template: str | None = None) -> dict | None:
        return self.reconstruct("cross", llm("cross"), [a, b])

    def energy(self, genes: dict) -> float | None:
        if genes.get("poison"):
            return None
        return (genes["x"] % 97) / 96.0

    def diversity_matrix(self, genes: dict) -> np.ndarray | None:
        if genes.get("no_diversity"):
            return None
        m = np.zeros((1, 8))
        m[0, genes["x"] % 8] = 1.0
        return m

    def integrity(self, genes: dict, rng: random.Random,
                  llm: LLMClient) -> dict | None:
        """Mock Step-6 integrity repair: evidenced by one host-client call;
        identity on the genes (nothing to repair in the toy)."""
        return self.reconstruct("integrity", llm("integrity"), [genes])

    def loci_view(self, genes: dict) -> dict:
        """Two named string loci for the canonical trace (RawSemanticGenome)."""
        return {"value": str(genes["x"]), "residue": str(genes["x"] % 97)}

    def endpoint_raw(self, genes: dict, bank: dict) -> "float | None":
        """E4 endpoint stand-in for the campaign runner's offline test: a
        deterministic function of the genes and the bank capacity (no
        bin-packing semantics; the mock has none)."""
        e = self.energy(genes)
        if e is None:
            return None
        return float(e * 2.0 + (bank.get("capacity", 100) / 1000.0))

    def validate(self, genes: dict) -> tuple[bool, str]:
        if not isinstance(genes.get("x"), int):
            return False, "x missing or not int"
        return True, "ok"
