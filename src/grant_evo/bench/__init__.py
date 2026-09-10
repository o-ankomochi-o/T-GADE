"""T-GADE benchmark harness.

Single production evolution loop (`core.BenchRun`) over problem adapters
(`adapter.ProblemAdapter`); selection delegates to grant_evo.tgade.selection.
Runners must use BenchRun and must not re-implement generation loops.
"""

from grant_evo.bench.adapter import Event, Individual, LLMClient, ProblemAdapter
from grant_evo.bench.core import BenchConfig, BenchRun, RunResult

__all__ = [
    "BenchConfig",
    "BenchRun",
    "Event",
    "Individual",
    "LLMClient",
    "ProblemAdapter",
    "RunResult",
]
