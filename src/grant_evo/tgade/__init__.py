"""T-GADE: thermodynamical selection of LLM-generated artifacts.

Submodules: free_energy (Gram free energy), selection (thermodynamical survivor
selection), engine (generational loop), embeddings (feature providers).
Submodules import their heavy dependencies lazily.
"""

from __future__ import annotations

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "DummyEmbedder",
    "GramFreeEnergy",
    "thermodynamical_select",
]


def __getattr__(name: str) -> object:
    """Lazy re-export so that submodule import cost is only paid on use."""
    if name in {"EmbeddingProvider", "OpenAIEmbedder", "DummyEmbedder"}:
        from grant_evo.tgade.embeddings import DummyEmbedder, EmbeddingProvider, OpenAIEmbedder

        return {"EmbeddingProvider": EmbeddingProvider, "OpenAIEmbedder": OpenAIEmbedder,
                "DummyEmbedder": DummyEmbedder}[name]
    if name == "GramFreeEnergy":
        from grant_evo.tgade.free_energy import GramFreeEnergy

        return GramFreeEnergy
    if name == "thermodynamical_select":
        from grant_evo.tgade.selection import thermodynamical_select

        return thermodynamical_select
    raise AttributeError(f"module 'grant_evo.tgade' has no attribute {name!r}")
