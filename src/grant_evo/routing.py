"""Single entry point for every LLM call.

Responsibilities:

- Resolve provider + model from ``config/models.yaml`` keys (e.g. ``claude_island.main``).
- Retry transient errors with tenacity.
- Emit one ``call`` lineage event per successful call (prompt/response digests,
  token usage, cost record id).
- Enforce per-call and cumulative cost caps via cost_tracker.

Other modules MUST import from here rather than calling SDKs directly.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from grant_evo.observability.cost_tracker import get_tracker
from grant_evo.observability.lineage import get_logger, text_digest


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cumulative_usd: float
    latency_seconds: float
    prompt_digest: str
    response_digest: str
    raw: dict[str, Any]


PROVIDER_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    # ``openai_compat`` reads the key from the per-entry ``api_key_env`` field
    # so multiple OpenAI-compatible providers (OpenRouter, Groq, Mistral, ...)
    # can coexist; see _openai_compat.
}


def _with_transport_limit(
    method: Callable[..., tuple[str, int, int, dict[str, Any]]],
    max_attempts: int,
) -> Callable[..., tuple[str, int, int, dict[str, Any]]]:
    """Clone a tenacity-wrapped provider call with a route-specific stop."""
    retry_with = getattr(method, "retry_with", None)
    if callable(retry_with):
        limited = cast(
            Callable[..., tuple[str, int, int, dict[str, Any]]],
            retry_with(stop=stop_after_attempt(max_attempts)),
        )
        bound_self = getattr(method, "__self__", None)
        if bound_self is not None:
            return cast(
                Callable[..., tuple[str, int, int, dict[str, Any]]],
                limited.__get__(bound_self, type(bound_self)),
            )
        return limited
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=2, max=10),
    )(method)


def _transport_attempt_count(method: Callable[..., Any]) -> int:
    """Read the completed attempt count across tenacity wrapper versions."""
    for owner in (getattr(method, "retry", None), method):
        statistics = getattr(owner, "statistics", None)
        if statistics is None:
            continue
        try:
            attempt_number = statistics.get("attempt_number")
        except AttributeError:
            continue
        if attempt_number is not None:
            return max(1, int(attempt_number))
    return 1


class Router:
    """LLM router that loads models config and dispatches to providers.

    The ``llm:`` section of the user configuration is the user-facing single source
    of truth for both API keys and the model pool. ``config/models.yaml`` is
    the framework default that ships with the repo.
    """

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or _find_repo_root()
        # override=True so a freshly rotated key in .env wins over a stale
        # shell-injected one.
        load_dotenv(self.repo_root / ".env", override=True)
        self.models_cfg = yaml.safe_load(
            (self.repo_root / "config" / "models.yaml").read_text(encoding="utf-8")
        )
        # Hydrate environment with API keys from applicant.yaml (does NOT
        # overwrite existing env vars — env wins so users who rotate via
        # shell still work).
        self.applicant = self._load_applicant()
        self._hydrate_api_keys()
        # Merge user-defined llm.models into a virtual ``llm_models`` namespace
        # so spec strings like ``llm_models.LLM1`` resolve via the same
        # ``resolve`` walk used by ``model_pool.*`` and legacy ``islands.*``.
        user_models = (self.applicant.get("llm") or {}).get("models") or {}
        if user_models:
            self.models_cfg.setdefault("llm_models", {}).update(user_models)
        self.tracker = get_tracker(self.repo_root)
        self.lineage = get_logger(self.repo_root)
        # WARN dedup set: emit each missing-field warning only once
        # per (spec, field) pair to avoid spamming stderr in long runs.
        self._warned_missing: set[tuple[str, str]] = set()

    def _warn_missing_field(self, spec: str, field: str, fallback: float | int) -> None:
        """Emit a one-shot WARN when a model entry lacks max_tokens / temperature．"""
        key = (spec, field)
        if key in self._warned_missing:
            return
        self._warned_missing.add(key)
        import sys as _sys
        _sys.stderr.write(
            f"[WARN] models.yaml: spec {spec!r} does not declare {field}; "
            f"using fallback={fallback}. Declare {field} explicitly in models.yaml.\n"
        )

    # ---------- applicant.yaml integration ----------

    def _load_applicant(self) -> dict[str, Any]:
        path = self.repo_root / "author" / "applicant.yaml"
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
        except yaml.YAMLError:
            return {}

    def _hydrate_api_keys(self) -> None:
        """Copy api_keys from applicant.yaml into os.environ if not already set.

        Reads ``llm.api_keys`` (new) with fallback to top-level ``api_keys``
        (legacy layout). Provider-name → env-var mapping comes from
        ``PROVIDER_ENV_VAR``. Free-form providers (e.g. ``openai_compat`` with
        ``api_key_env``) are honoured via the entry's own field at call time.
        """
        keys = (self.applicant.get("llm") or {}).get("api_keys") or self.applicant.get(
            "api_keys"
        ) or {}
        if not isinstance(keys, dict):
            return
        for provider, key in keys.items():
            if not key:
                continue
            if str(provider).lower() == "gemini":
                raise ValueError(
                    "llm.api_keys.gemini is no longer supported; configure "
                    "OPENROUTER_API_KEY via llm.api_keys.openrouter or .env"
                )
            env_var = PROVIDER_ENV_VAR.get(
                provider, f"{str(provider).upper()}_API_KEY"
            )
            os.environ.setdefault(env_var, str(key))

    # ---------- spec resolution ----------

    def resolve(self, spec: str) -> tuple[str, str, dict[str, Any]]:
        """Resolve a config-spec into (provider, model, options).

        Accepted shapes:
        - ``islands.<island>.main`` / ``.sub`` — uses ``main_model`` / ``sub_model``.
        - ``judges.<key>`` / ``operators.<key>`` — uses the ``model`` field.
        """
        parts = spec.split(".")

        # Shorthand: islands.<name>.main|sub
        if len(parts) >= 2 and parts[-1] in {"main", "sub"}:
            parent: Any = self.models_cfg
            for p in parts[:-1]:
                if not isinstance(parent, dict) or p not in parent:
                    raise KeyError(
                        f"models.yaml: cannot resolve {spec!r} at part {p!r}"
                    )
                parent = parent[p]
            mkey = "main_model" if parts[-1] == "main" else "sub_model"
            if not isinstance(parent, dict) or mkey not in parent:
                raise KeyError(
                    f"models.yaml: {spec!r} expects {mkey!r} under "
                    f"{'.'.join(parts[:-1])}"
                )
            return parent["provider"], parent[mkey], parent

        # General path: walk all parts, expect a dict with provider+model.
        node: Any = self.models_cfg
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"models.yaml: cannot resolve {spec!r} at part {p!r}")
            node = node[p]
        if isinstance(node, dict) and "model" in node and "provider" in node:
            return node["provider"], node["model"], node
        raise KeyError(f"models.yaml: cannot interpret {spec!r}")

    # ---------- public call ----------

    def call(
        self,
        spec: str,
        *,
        prompt: str,
        system: str | None = None,
        purpose: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        provider, model, opts = self.resolve(spec)
        if provider == "gemini":
            raise ValueError(
                "provider 'gemini' is no longer supported; route Gemini through "
                "OpenRouter with provider='openai_compat', "
                "base_url='https://openrouter.ai/api/v1', and "
                "api_key_env='OPENROUTER_API_KEY'"
            )
        if extra_body and provider != "openai_compat":
            raise ValueError("extra_body is supported only for openai_compat routes")
        # max_tokens / temperature should be declared by the model_pool entry;
        # otherwise the fallback (2048 / 0.7) is used and a single WARN is
        # written to stderr so that silent truncation stays visible.
        if max_tokens is None and "max_tokens" not in opts:
            self._warn_missing_field(spec, "max_tokens", 2048)
        if temperature is None and "temperature" not in opts:
            self._warn_missing_field(spec, "temperature", 0.7)
        max_tokens = max_tokens if max_tokens is not None else int(opts.get("max_tokens", 2048))
        temperature = (
            temperature if temperature is not None else float(opts.get("temperature", 0.7))
        )
        reasoning_effort = (
            str(opts["reasoning_effort"]) if opts.get("reasoning_effort") else None
        )
        reasoning_mode = str(opts.get("reasoning_mode", "standard"))
        timeout_seconds = float(opts.get("timeout_seconds", 300.0))
        max_transport_attempts = int(opts.get("max_transport_attempts", 3))
        if max_transport_attempts < 1:
            raise ValueError(
                f"max_transport_attempts must be positive for {spec!r}, "
                f"got {max_transport_attempts}"
            )

        prompt_digest = text_digest(prompt)
        started = time.perf_counter()
        if provider == "anthropic":
            transport_method = _with_transport_limit(
                self._anthropic, max_transport_attempts
            )
            text, in_tok, out_tok, raw = transport_method(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif provider == "openai":
            transport_method = _with_transport_limit(
                self._openai, max_transport_attempts
            )
            text, in_tok, out_tok, raw = transport_method(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
            )
        elif provider == "deepseek":
            # OpenAI-compatible API at api.deepseek.com.
            transport_method = _with_transport_limit(
                self._openai_compat, max_transport_attempts
            )
            text, in_tok, out_tok, raw = transport_method(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url="https://api.deepseek.com/v1",
                api_key_env="DEEPSEEK_API_KEY",
                reasoning_effort=None,
                timeout_seconds=timeout_seconds,
                extra_body=None,
            )
        elif provider == "openai_compat":
            # Generic OpenAI-compatible (OpenRouter, Groq, Mistral, ...).
            base_url = opts.get("base_url")
            api_key_env = opts.get("api_key_env", "OPENAI_API_KEY")
            if not base_url:
                raise ValueError(
                    f"openai_compat entry must specify ``base_url``: {spec!r}"
                )
            transport_method = _with_transport_limit(
                self._openai_compat, max_transport_attempts
            )
            text, in_tok, out_tok, raw = transport_method(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=base_url,
                api_key_env=api_key_env,
                reasoning_effort=reasoning_effort,
                timeout_seconds=timeout_seconds,
                extra_body=extra_body,
            )
        else:
            raise ValueError(f"unknown provider {provider!r}")
        latency_seconds = time.perf_counter() - started
        transport_attempts = _transport_attempt_count(transport_method)
        provider_reported_cost = _optional_float(raw.get("provider_reported_cost_usd"))
        if opts.get("require_provider_reported_cost") and provider_reported_cost is None:
            raise RuntimeError(
                f"route {spec!r} requires provider-reported cost, but none was returned"
            )
        raw.setdefault("reasoning_mode", reasoning_mode)
        raw.setdefault("reasoning_effort", reasoning_effort)
        raw.setdefault("latency_seconds", round(latency_seconds, 6))
        raw.setdefault("transport_attempts", transport_attempts)

        cost_rec = self.tracker.record(
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            purpose=purpose,
            reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
            cached_input_tokens=int(raw.get("cached_input_tokens", 0) or 0),
            cache_write_tokens=int(raw.get("cache_write_tokens", 0) or 0),
            latency_seconds=latency_seconds,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            provider_reported_cost_usd=provider_reported_cost,
            transport_attempts=transport_attempts,
        )
        raw["estimated_cost_usd"] = cost_rec.estimated_total_usd
        raw["provider_reported_cost_usd"] = cost_rec.provider_reported_cost_usd
        raw["ledger_cost_usd"] = cost_rec.total_usd
        raw["cost_source"] = cost_rec.cost_source
        response_digest = text_digest(text)
        self.lineage.log(
            "call",
            provider=provider,
            model=model,
            spec=spec,
            purpose=purpose,
            prompt_digest=prompt_digest,
            response_digest=response_digest,
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning_tokens=cost_rec.reasoning_tokens,
            cached_input_tokens=cost_rec.cached_input_tokens,
            cache_write_tokens=cost_rec.cache_write_tokens,
            latency_seconds=cost_rec.latency_seconds,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            estimated_cost_usd=cost_rec.estimated_total_usd,
            provider_reported_cost_usd=cost_rec.provider_reported_cost_usd,
            cost_source=cost_rec.cost_source,
            transport_attempts=cost_rec.transport_attempts,
            cost_usd=cost_rec.total_usd,
            cumulative_usd=cost_rec.cumulative_usd,
        )
        return LLMResponse(
            text=text,
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost_rec.total_usd,
            cumulative_usd=cost_rec.cumulative_usd,
            latency_seconds=latency_seconds,
            prompt_digest=prompt_digest,
            response_digest=response_digest,
            raw=raw,
        )

    # ---------- providers ----------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _anthropic(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, int, int, dict[str, Any]]:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


        # ``temperature`` parameter. Detect and omit when applicable.
        is_reasoning = "opus-4-7" in model or "opus-4-8" in model
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        msg = client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        usage = msg.usage
        in_tok = int(getattr(usage, "input_tokens", 0))
        out_tok = int(getattr(usage, "output_tokens", 0))
        raw = {"id": msg.id, "stop_reason": msg.stop_reason, "is_reasoning": is_reasoning}
        return text, in_tok, out_tok, raw

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _openai(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        timeout_seconds: float,
    ) -> tuple[str, int, int, dict[str, Any]]:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=timeout_seconds,
            max_retries=0,
        )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # GPT-5 reasoning family rejects non-default temperature; reasoning
        # tokens are emitted regardless. We keep max_completion_tokens generous
        # so the visible answer survives reasoning-token consumption.
        is_reasoning = model.startswith("gpt-5") or model.startswith("o")
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        if not is_reasoning:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0))
        out_tok = int(getattr(usage, "completion_tokens", 0))
        raw = {
            "id": resp.id,
            "finish_reason": choice.finish_reason,
            "is_reasoning": is_reasoning,
            **_usage_token_details(usage),
        }
        return text, in_tok, out_tok, raw

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _openai_compat(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        base_url: str,
        api_key_env: str,
        reasoning_effort: str | None,
        timeout_seconds: float,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, int, int, dict[str, Any]]:
        """Generic OpenAI-compatible HTTP API.

        Used for DeepSeek, OpenRouter, Groq, Mistral, Together, etc. The
        OpenAI Python client supports a custom ``base_url`` so we just point
        it at the right endpoint and use a different env var for the key.

        OpenRouter quirk: the model id is namespaced (``openai/gpt-5``,
        ``anthropic/claude-opus-4-6``, ``google/gemini-3.1-pro-preview``).
        Reasoning models (gpt-5 family / claude-opus-4-7+ / o-series) reject
        non-default ``temperature`` regardless of which gateway proxies them,
        so detect via the namespaced suffix and strip when needed.
        """
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Configure in applicant.yaml "
                f"``llm.api_keys`` (key matches provider name) or .env."
            )

        # Reasoning detection covers both bare model ids (e.g. "gpt-5")
        # and namespaced OpenRouter ids (e.g. "openai/gpt-5").
        bare = model.rsplit("/", 1)[-1] if "/" in model else model
        is_reasoning = (
            bare.startswith("gpt-5")
            or bare.startswith("o1")
            or bare.startswith("o3")
            or bare.startswith("o4")
            or "opus-4-7" in bare
            or "opus-4-8" in bare
        )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        if not is_reasoning:
            kwargs["temperature"] = temperature
        is_openrouter = "openrouter.ai" in base_url.lower()
        request_extra_body = dict(extra_body or {})
        if reasoning_effort:
            if is_openrouter:
                reasoning = dict(request_extra_body.get("reasoning") or {})
                reasoning["effort"] = reasoning_effort
                request_extra_body["reasoning"] = reasoning
            else:
                kwargs["reasoning_effort"] = reasoning_effort
        if request_extra_body:
            kwargs["extra_body"] = request_extra_body
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        in_tok = int(getattr(usage, "prompt_tokens", 0))
        out_tok = int(getattr(usage, "completion_tokens", 0))
        raw = {
            "id": resp.id,
            "response_model": getattr(resp, "model", None),
            "finish_reason": choice.finish_reason,
            "base_url": base_url,
            "is_reasoning": is_reasoning,
            "usage": usage.model_dump() if usage is not None else {},
            **_usage_token_details(usage),
            **_usage_cost_details(usage),
        }
        return text, in_tok, out_tok, raw


# ---------- module-level helper ----------

_router: Router | None = None


def get_router(repo_root: Path | None = None) -> Router:
    global _router
    if _router is None:
        _router = Router(repo_root=repo_root)
    return _router


def reset_router_for_tests() -> None:
    global _router
    _router = None


def _find_repo_root() -> Path:
    here = Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "models.yaml").exists():
            return candidate
    raise RuntimeError("cannot locate repo root (no config/models.yaml found upward)")


def _usage_token_details(usage: Any) -> dict[str, int]:
    """Extract optional reasoning/cache counters from OpenAI-style usage."""
    if usage is None:
        return {
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
        }
    data = usage.model_dump() if hasattr(usage, "model_dump") else {}
    completion = data.get("completion_tokens_details") or {}
    prompt = data.get("prompt_tokens_details") or {}
    return {
        "reasoning_tokens": int(completion.get("reasoning_tokens") or 0),
        "cached_input_tokens": int(prompt.get("cached_tokens") or 0),
        "cache_write_tokens": int(
            prompt.get("cache_write_tokens")
            or data.get("cache_write_tokens")
            or 0
        ),
    }


def _usage_cost_details(usage: Any) -> dict[str, float | None]:
    """Extract OpenRouter billed cost and optional upstream cost."""
    data = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
    cost_details = data.get("cost_details") or {}
    return {
        "provider_reported_cost_usd": _optional_float(data.get("cost")),
        "upstream_inference_cost_usd": _optional_float(
            cost_details.get("upstream_inference_cost")
        ),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
