"""Section-text embedding providers for T-GADE diversity calculation.

The free-energy term (UpdatePlan §1, §8) needs per-section embeddings to
form Gram matrices L_k. We default to OpenAI ``text-embedding-3-small``
(1536-dim, multilingual, very cheap) because the project already requires
the openai SDK and the user has an OPENROUTER_API_KEY which routes most
calls but the OpenAI SDK uses OPENAI_API_KEY directly.

Pluggable design:

- ``EmbeddingProvider``: protocol — ``embed(texts) -> ndarray``
- ``OpenAIEmbedder``: real provider, default for production runs
- ``DummyEmbedder``: deterministic hash-based pseudo-embedding for tests
  (no network, ~10 ms for 100 texts; not semantically meaningful)

Embedding vectors are L2-normalised so Gram entries are cosine similarities
in [-1, 1]. This keeps logdet finite under reasonable ε regularisation.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

import numpy as np

from grant_evo.observability.cost_tracker import get_tracker


class EmbeddingProvider(Protocol):
    """Embed a batch of texts to a (N, D) float32 array. Rows L2-normalised."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    """OpenAI ``text-embedding-3-small`` (1536-dim, multilingual).

    Cost: $0.02 per 1M input tokens. For T-GADE with N=17 candidates, M=5
    sections, ~600 chars per section ≈ 400 tokens → 17 × 5 × 400 = 34k
    tokens per re-embed = $0.0007. Negligible.

    Routing options (resolved in order):

    1. Explicit ``base_url`` + ``api_key_env`` — caller knows what they want.
    2. Direct OpenAI: ``OPENAI_API_KEY`` set and not obviously a routing key.
    3. OpenRouter fallback: ``OPENROUTER_API_KEY`` set → use the OpenRouter
       base url (the cheap-pool config already routes everything via
       OpenRouter).

    The constructor loads .env via python-dotenv so it works whether or not
    the calling shell has the keys exported.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key_env: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        prefer_openrouter: bool | None = None,
        track_cost: bool = False,
        purpose: str | None = None,
    ) -> None:
        self.model = model
        if dimensions is not None and dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.dimensions = dimensions
        self.track_cost = track_cost
        self.purpose = purpose or "embedding"
        self.total_input_tokens = 0
        self.total_cost_usd = 0.0

        # Load .env so freshly-rotated keys are picked up even when the
        # shell hasn't exported them.
        try:
            from dotenv import load_dotenv

            from grant_evo.routing import _find_repo_root

            load_dotenv(_find_repo_root() / ".env", override=False)
        except Exception:
            # _find_repo_root may not work outside of a project context;
            # silently fall through to bare environ.
            pass

        api_key, resolved_base_url = self._resolve_auth(
            api_key_env=api_key_env,
            base_url=base_url,
            prefer_openrouter=prefer_openrouter,
        )

        # Lazy import: openai SDK present but optional in dev.
        from openai import OpenAI

        if resolved_base_url:
            self._client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        else:
            self._client = OpenAI(api_key=api_key)
        self._base_url = resolved_base_url

        # Probe dimensions if not specified.
        if dimensions is None:
            self.dim = 1536  # default for text-embedding-3-small
        else:
            self.dim = dimensions

    @staticmethod
    def _resolve_auth(
        *,
        api_key_env: str | None,
        base_url: str | None,
        prefer_openrouter: bool | None,
    ) -> tuple[str, str | None]:
        """Pick the best auth path based on what's available.

        Returns (api_key, base_url_or_None).
        """
        # (1) Explicit override.
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(
                    f"OpenAIEmbedder: {api_key_env} not in env"
                )
            return key, base_url

        openai_key = os.environ.get("OPENAI_API_KEY")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")

        if prefer_openrouter and openrouter_key:
            return openrouter_key, OpenAIEmbedder.OPENROUTER_BASE_URL

        # (2) Try direct OpenAI first if available.
        if openai_key:
            return openai_key, base_url

        # (3) Fallback to OpenRouter.
        if openrouter_key:
            return openrouter_key, OpenAIEmbedder.OPENROUTER_BASE_URL

        raise RuntimeError(
            "OpenAIEmbedder needs either OPENAI_API_KEY or OPENROUTER_API_KEY in env"
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), self.dim) L2-normalised float32 array.

        Empty / whitespace-only inputs are mapped to zero vectors. They are
        filtered out before the API call (OpenAI's embeddings endpoint
        rejects empty strings with HTTP 400) and re-inserted as zero rows
        in the returned tensor.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Index map: which output rows correspond to which API inputs.
        api_inputs: list[str] = []
        api_index: list[int] = []  # position in `texts` for each api_input
        for i, t in enumerate(texts):
            if t and t.strip():
                api_inputs.append(t)
                api_index.append(i)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if not api_inputs:
            return out

        if self.dimensions is not None:
            resp = self._client.embeddings.create(
                model=self.model,
                input=api_inputs,
                dimensions=self.dimensions,
            )
        else:
            resp = self._client.embeddings.create(model=self.model, input=api_inputs)
        if not resp.data:
            raise RuntimeError("embedding endpoint returned no data")
        if self.track_cost:
            usage = getattr(resp, "usage", None)
            input_tokens = int(
                getattr(usage, "prompt_tokens", 0)
                or getattr(usage, "total_tokens", 0)
                or 0
            )
            if input_tokens <= 0:
                raise RuntimeError("embedding cost tracking requires usage tokens")
            provider = "openai" if self._base_url is None else "openai_compat"
            ledger_model = self.model
            if provider == "openai_compat" and "/" not in ledger_model:
                ledger_model = f"openai/{ledger_model}"
            record = get_tracker().record(
                provider=provider,
                model=ledger_model,
                input_tokens=input_tokens,
                output_tokens=0,
                purpose=self.purpose,
            )
            self.total_input_tokens += input_tokens
            self.total_cost_usd += record.total_usd
        api_vecs = np.asarray(
            [d.embedding for d in resp.data], dtype=np.float32
        )
        if api_vecs.shape != (len(api_inputs), self.dim):
            raise RuntimeError(
                f"embedding shape mismatch: expected {(len(api_inputs), self.dim)}, "
                f"got {api_vecs.shape}"
            )
        if not np.isfinite(api_vecs).all():
            raise RuntimeError("embedding endpoint returned non-finite values")
        # Defensive renormalisation; empty rows already zeros.
        norms = np.linalg.norm(api_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        api_vecs = api_vecs / norms
        for src_pos, dst_pos in enumerate(api_index):
            out[dst_pos] = api_vecs[src_pos]
        return out


class DummyEmbedder:
    """Deterministic non-semantic embedder for tests.

    Maps each text to a unit vector via SHA-256 → bytes → bipolar floats.
    Two identical texts yield identical embeddings (good for round-trip
    tests of the Gram update). Two different texts have nearly orthogonal
    embeddings on average (good enough for diversity-monotonicity checks).
    """

    def __init__(self, *, dim: int = 64) -> None:
        if dim <= 0 or dim % 8 != 0:
            raise ValueError(f"dim must be positive multiple of 8, got {dim}")
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        bytes_per_vec = self.dim // 8
        for i, t in enumerate(texts):
            if not t.strip():
                continue
            # Hash repeatedly to get enough bytes. SHA-256 → 32 bytes per round.
            material = b""
            seed = t.encode("utf-8", errors="replace")
            rounds = (bytes_per_vec + 31) // 32
            for r in range(rounds):
                h = hashlib.sha256(seed + str(r).encode()).digest()
                material += h
            material = material[:bytes_per_vec]
            # 1 byte → 8 bipolar floats. Bit set → +1, unset → -1.
            unpacked = np.unpackbits(np.frombuffer(material, dtype=np.uint8))
            bits = unpacked[: self.dim].astype(np.float32) * 2.0 - 1.0
            n = math.sqrt(self.dim)
            out[i] = bits / n
        return out
