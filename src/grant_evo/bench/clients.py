"""Real LLM clients for the bench engine (G2c E3).

OpenRouterClient returns LLMResult with typed usage (prompt/completion tokens,
cost_usd, provider, request_id, retries) so the engine's call ledger and
manifest totals are complete (T24). A hard budget cap raises BudgetExceeded
BEFORE the call that would cross it — fail-closed spending.

The API key is read from the repo .env at call time and never logged
(reference_openrouter_env rule). Transport is injectable for offline tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import urllib.request
from pathlib import Path

from grant_evo.tgade.engine import LLMResult

_REPO = Path(__file__).resolve().parents[3]


class BudgetExceeded(RuntimeError):
    """Hard-cap refusal. Carries `usage` = {"charged_usd", "attempts", "error"}
    for the LOGICAL call being refused, so every caller (engine, MULTISTART,
    EoH shim) can persist the cost already incurred by earlier physical
    attempts of that call."""

    usage: dict = {}


def _read_key() -> str:
    for line in (_REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")


def _default_transport(payload: dict, key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenRouterClient:
    """Callable LLMClient: prompt -> LLMResult (text=None on failure).

    HARD cap semantics: the per-attempt
    reservation is DERIVED mechanically, not asserted. For byte-level BPE
    tokenizers (the qwen3 family included) every token spans >= 1 UTF-8
    byte, so input tokens <= utf8_bytes(prompt); provider chat-template
    overhead is covered by a documented conservative constant:
        tokens_in_upper = utf8_bytes(prompt) + chat_overhead_tokens
        reserve = price_in_usd_per_m * tokens_in_upper/1e6
                + price_out_usd_per_m * max_tokens/1e6
    with PINNED ceiling prices. Prompts longer than max_input_bytes
    (UTF-8 bytes, not code points) are refused BEFORE transport. An attempt is refused when
    spent + reserve > budget_usd. Every attempt (retries included) is
    accounted in attempt_log and spent_usd; an attempt whose cost the
    provider does not report (or that dies in transport) is charged the full
    reservation (conservative over-count, never under). A reported cost
    above the reservation means the pinned prices were violated: the client
    POISONS itself (all further calls refused). max_tokens is always sent.
    """

    def __init__(self, model: str, *, temperature: float = 1.0,
                 budget_usd: float = 1.0, timeout: int = 120, retries: int = 0,
                 max_tokens: int = 2048, price_in_usd_per_m: float = 1.0,
                 price_out_usd_per_m: float = 2.0, max_input_bytes: int = 60000,
                 chat_overhead_tokens: int = 512, seed: "int | None" = None,
                 global_budget=None, min_interval_s: float = 0.0,
                 provider_order: "list | None" = None, allow_fallbacks: bool = True,
                 usage_accounting: bool = True, transport=None):
        def _num(name, v, *, integer=False, minimum=0.0):
            # bool is an int subclass; reject it and non-finite/negative
            # values at construction.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a number, got {v!r}")
            if integer and not isinstance(v, int):
                raise ValueError(f"{name} must be an int, got {v!r}")
            if not math.isfinite(v) or v < minimum:
                raise ValueError(f"{name} must be finite and >= {minimum}, got {v!r}")
            return v

        _num("budget_usd", budget_usd)
        _num("temperature", temperature)
        _num("price_in_usd_per_m", price_in_usd_per_m)
        _num("price_out_usd_per_m", price_out_usd_per_m)
        _num("max_tokens", max_tokens, integer=True, minimum=1)
        _num("max_input_bytes", max_input_bytes, integer=True, minimum=1)
        _num("chat_overhead_tokens", chat_overhead_tokens, integer=True)
        _num("retries", retries, integer=True)
        if seed is not None:
            _num("seed", seed, integer=True)
        self.model = model
        self.temperature = temperature
        self.budget_usd = budget_usd
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens
        self.price_in_usd_per_m = price_in_usd_per_m
        self.price_out_usd_per_m = price_out_usd_per_m
        self.max_input_bytes = max_input_bytes
        self.chat_overhead_tokens = chat_overhead_tokens
        self.seed = seed  # BASE for the per-call seed schedule (see below)
        self.usage_accounting = usage_accounting
        self.spent_usd = 0.0
        self.calls = 0
        self.attempt_log: list[dict] = []
        self._poisoned: "str | None" = None  # reservation breach = stop all
        self._global = global_budget  # cross-run authority
        _num("min_interval_s", min_interval_s)
        self.min_interval_s = float(min_interval_s)
        self._last_call_ts = 0.0
        # provider pinning. OpenRouter routing block:
        # {"order": [...], "allow_fallbacks": false} => one provider only.
        self.provider_order = list(provider_order) if provider_order else None
        self.allow_fallbacks = bool(allow_fallbacks)
        self._transport = transport or _default_transport
        # Admission lock (rate slot + budget incl. in-flight reservations),
        # per-call attempt attribution, seed from the logical call id when the engine passes one.
        self._lock = threading.Lock()
        self._inflight_usd = 0.0
        self.accepts_call_id = True

    def _budget_exceeded(self, msg: str, attempts: int, call_charged: float, attempts_list=None) -> BudgetExceeded:
        exc = BudgetExceeded(msg)
        exc.usage = {"charged_usd": float(call_charged), "error": msg,
                     "attempts": list(attempts_list if attempts_list is not None else (self.attempt_log[-attempts:] if attempts else []))}
        return exc

    def _charge(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd

    def _reserve(self, prompt: str) -> float:
        tokens_in_upper = (len(prompt.encode("utf-8"))
                           + self.chat_overhead_tokens)
        return (self.price_in_usd_per_m * tokens_in_upper / 1e6
                + self.price_out_usd_per_m * self.max_tokens / 1e6)

    def __call__(self, prompt: str, **opts) -> LLMResult:
        prompt = str(prompt)
        nbytes = len(prompt.encode("utf-8"))
        if nbytes > self.max_input_bytes:
            raise ValueError(
                f"prompt {nbytes} UTF-8 bytes exceeds max_input_bytes "
                f"{self.max_input_bytes} (refused before transport)")
        reserve = self._reserve(prompt)
        temperature = opts.get("temperature", self.temperature)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature < 0):
            raise ValueError(f"per-call temperature must be finite and >= 0, got {temperature!r}")
        unknown = set(opts) - {"temperature", "call_id"}
        call_id = opts.get("call_id")
        if unknown:
            raise ValueError(f"unsupported per-call options: {sorted(unknown)}")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": float(temperature),
            "max_tokens": self.max_tokens,
        }
        if self.provider_order is not None:
            payload["provider"] = {"order": list(self.provider_order),
                                   "allow_fallbacks": self.allow_fallbacks}
        if self.seed is not None:
            # Deterministic PER-CALL schedule: a constant
            # provider seed would collapse repeated identical prompts (e.g.
            # MULTISTART i1 draws) to one completion. seed_sent = frozen
            # hash of (base, logical call index); recorded per attempt.
            # a logical call id from the engine gives a schedule-independent seed;
            # without one the legacy completion-order counter is kept.
            logical = call_id if call_id is not None else self.calls
            seed_sent = int.from_bytes(hashlib.sha256(
                f"{self.seed}:{logical}".encode("utf-8")).digest()[:4], "big")
            payload["seed"] = seed_sent
        if self.usage_accounting:
            payload["usage"] = {"include": True}
        key = _read_key()
        attempts = 0
        last_err = None
        call_charged = 0.0
        last_usage: dict = {}
        empty_content = False
        my_attempts: list = []
        held = False
        try:
          while attempts <= self.retries:
            # Admission (atomic): take the next rate slot, then check the cap against
            # spent + in-flight reservations + this reservation; hold this call's
            # reservation until the logical call ends. Sleeping happens OUTSIDE the lock.
            with self._lock:
                if self.min_interval_s > 0:
                    slot = max(self._last_call_ts + self.min_interval_s, time.time())
                    wait = slot - time.time()
                    self._last_call_ts = slot
                else:
                    wait = 0.0
                    self._last_call_ts = time.time()
                if self._poisoned:
                    raise self._budget_exceeded(self._poisoned, attempts, call_charged, my_attempts)
                # RESERVE before transport: the cap can never be crossed even by
                # a maximally expensive attempt (mechanical bound).
                others = self._inflight_usd - (reserve if held else 0.0)
                if self.spent_usd + others + reserve > self.budget_usd:
                    raise self._budget_exceeded(
                        f"budget cap {self.budget_usd} USD: spent {self.spent_usd:.4f}"
                        f" + in-flight {others:.6f} + reserve {reserve:.6f} would exceed it", attempts, call_charged, my_attempts)
                if not held:
                    self._inflight_usd += reserve
                    held = True
            if wait > 0:
                time.sleep(wait)
            reservation_id = None
            if self._global is not None:
                # Durable before transport: a process death cannot erase an
                # already-incurred campaign charge from the global cap.
                reservation_id = self._global.reserve(
                    reserve, f"{self.model} attempt")
            attempts += 1
            self.calls += 1
            try:
                data = self._transport(payload, key, self.timeout)
            except Exception as exc:  # noqa: BLE001 - recorded, retried
                last_err = f"{type(exc).__name__}: {exc}"
                # cost of a dead attempt is unrecoverable: charge the FULL
                # reservation (conservative over-count, never under).
                self._charge(reserve)
                call_charged += reserve
                if self._global is not None:
                    self._global.settle(
                        reservation_id, reserve, "failed-attempt reservation")
                _att = {"ok": False, "error": last_err,
                        "charged_usd": reserve,
                        "seed_sent": payload.get("seed")}
                self.attempt_log.append(_att); my_attempts.append(_att)
                time.sleep(min(2 ** attempts, 8))
                continue
            choices = data.get("choices") or []
            if not choices and isinstance(data, dict) and (data.get("error") or not data):
                # Provider/gateway ERROR BODY returned with HTTP 200 (rate
                # limit, overload, upstream failure): a TRANSPORT-class
                # failure, retryable within the bounded policy. Charged at
                # the full reservation (no usage reported). E3 run 152626:
                # 15/44 diagnostic calls were this case, not empty content.
                err = data.get("error") if isinstance(data, dict) else None
                code = (err or {}).get("code") if isinstance(err, dict) else None
                last_err = f"provider error body (code={code!r}): {str(err)[:160]}"
                self._charge(reserve)
                call_charged += reserve
                if self._global is not None:
                    self._global.settle(reservation_id, reserve, "provider-error reservation")
                _att = {"ok": False, "error": last_err,
                        "charged_usd": reserve,
                        "seed_sent": payload.get("seed"),
                        "request_id": data.get("id") if isinstance(data, dict) else None}
                self.attempt_log.append(_att); my_attempts.append(_att)
                if attempts <= self.retries:
                    time.sleep(min(2 ** attempts, 8))
                continue
            text = (choices[0].get("message", {}).get("content")
                    if choices else None)
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            # cost must be a FINITE NON-NEGATIVE number ( NaN
            # disables every later comparison; negative refunds the cap).
            # Absent cost => charge the full reservation. Present-but-
            # malformed cost => charge the full reservation AND poison.
            cost_valid = (isinstance(cost, (int, float))
                          and math.isfinite(cost) and cost >= 0)
            charged = float(cost) if cost_valid else reserve
            self._charge(charged)
            call_charged += charged
            if self._global is not None:
                self._global.settle(reservation_id, charged, "attempt")
            if cost is not None and not cost_valid:
                self._poisoned = (
                    f"non-finite/negative usage.cost {cost!r}: charged the "
                    f"full reservation; client poisoned")
            if charged > reserve:
                # the reservation was NOT a true upper bound: record the true
                # charge, then refuse every further call (fail-closed).
                self._poisoned = (
                    f"reservation breached: attempt cost {charged:.6f} USD > "
                    f"reserve {reserve:.6f} USD (pinned prices violated); "
                    f"client poisoned")
            usable = bool(text)
            last_err = (None if usable else
                        f"empty content (finish={choices[0].get('finish_reason') if choices else 'no-choice'})")
            if not usable:
                # EMPTY CONTENT is a model outcome, not a transport failure:
                # never retried (it would bias the sample toward lucky
                # completions); it is a vacancy-class result upstream.
                empty_content = True
            else:
                empty_content = False
            if (self.provider_order is not None and not self.allow_fallbacks
                    and data.get("provider") not in self.provider_order):
                # routing contract violated: never silently accept another
                # provider's output; charged, then poisoned.
                self._poisoned = (f"provider routing violated: got {data.get('provider')!r},"
                                  f" pinned {self.provider_order}; client poisoned")
            _att = {
                "ok": usable, "error": last_err,
                "charged_usd": charged,
                "provider": data.get("provider"),
                "provider_reported_cost_usd": (float(cost)
                                                if cost_valid else None),
                "seed_sent": payload.get("seed"),
                "request_id": data.get("id"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
            self.attempt_log.append(_att); my_attempts.append(_att)
            last_usage = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                # The evidence ledger must match the amount charged by this
                # client, including failed retries and conservative reserves.
                "cost_usd": call_charged,
                "provider_reported_cost_usd": (float(cost)
                                                if cost_valid else None),
                "provider": data.get("provider"),
                "model": data.get("model"),
                "request_id": data.get("id"),
                "retries": attempts - 1,

                "finish_reason": (choices[0].get("finish_reason") if choices else None),
            }
            last_usage["attempts"] = list(my_attempts)
            if text:
                return LLMResult(text=text, usage=last_usage)
            if empty_content:
                break  # model outcome: no retry
        finally:
            if held:
                with self._lock:
                    self._inflight_usd -= reserve
        last_usage.update({"retries": attempts - 1, "cost_usd": call_charged,
                           "error": last_err,
                           "attempts": list(my_attempts) if attempts else []})
        return LLMResult(text=None, usage=last_usage)
