"""Compatibility shim: the production loop lives in grant_evo.tgade.engine
. bench keeps only adapters, mocks, and orchestration."""

from grant_evo.tgade.engine import (  # noqa: F401
    BenchConfig,
    BenchRun,
    PopulationExtinctionError,
    RunResult,
)

__all__ = ["BenchConfig", "BenchRun", "PopulationExtinctionError", "RunResult"]
