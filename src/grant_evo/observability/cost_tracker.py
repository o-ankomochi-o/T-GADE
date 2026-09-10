"""Track cumulative LLM cost.

Reads thresholds from ``config/budgets.yaml`` and per-token prices from
``config/prices.yaml``. Appends one JSON line per call to
``outputs/cost_ledger.jsonl``. Raises ``BudgetExceeded`` when the hard cap is
hit so the engine cannot silently overspend.
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

_LOCK = threading.Lock()
_TOTAL_CAP_ENV = "T_GADE_TOTAL_HARD_CAP"
_TOTAL_CAP_OPT_IN_ENV = "T_GADE_TOTAL_HARD_CAP_OPT_IN"


class BudgetExceeded(RuntimeError):  # noqa: N818 - public API is established
    """Raised when a planned call would push cumulative cost over the hard cap."""


@dataclass
class CostRecord:
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_usd: float
    output_usd: float
    total_usd: float
    cumulative_usd: float
    purpose: str | None = None  # e.g. "init_population:claude_island"
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    latency_seconds: float | None = None
    reasoning_mode: str | None = None
    reasoning_effort: str | None = None
    estimated_total_usd: float | None = None
    provider_reported_cost_usd: float | None = None
    provider_cost_rejected: bool = False
    cost_source: str = "price_table"
    transport_attempts: int = 1


class CostTracker:
    """Process-singleton-ish cost tracker."""

    def __init__(
        self,
        budgets_path: Path,
        prices_path: Path,
        ledger_path: Path,
    ) -> None:
        self.budgets = yaml.safe_load(budgets_path.read_text(encoding="utf-8")) or {}
        self.prices = yaml.safe_load(prices_path.read_text(encoding="utf-8")) or {}
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.cumulative = self._read_cumulative()
        # Per-run baseline: a parent process may export ``T_GADE_RUN_BASELINE_USD``
        # (the cumulative spend at run start); ``run_cost`` / ``run_hard_cap`` are
        # then checked against ``cumulative - baseline`` so that earlier runs do
        # not exhaust the cap of the current run.
        self.run_baseline_usd = _read_run_baseline_env(self.cumulative)

    # ---------- public API ----------

    def price_for(self, provider: str, model: str) -> tuple[float, float]:
        per_provider = self.prices.get(provider, {}) or {}
        if model in per_provider:
            return float(per_provider[model]["input"]), float(
                per_provider[model]["output"]
            )
        defaults = self.prices.get("defaults", {})
        return float(defaults.get("input", 5.0)), float(defaults.get("output", 25.0))

    def record(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str | None = None,
        reasoning_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
        latency_seconds: float | None = None,
        reasoning_mode: str | None = None,
        reasoning_effort: str | None = None,
        provider_reported_cost_usd: float | None = None,
        transport_attempts: int = 1,
    ) -> CostRecord:
        in_price, out_price = self.price_for(provider, model)
        in_usd = (input_tokens / 1_000_000.0) * in_price
        out_usd = (output_tokens / 1_000_000.0) * out_price
        estimated_total = in_usd + out_usd
        reported_cost: float | None = None
        provider_cost_rejected = False
        if provider_reported_cost_usd is not None:
            candidate = float(provider_reported_cost_usd)
            if math.isfinite(candidate) and candidate >= 0.0:
                reported_cost = candidate
            else:
                provider_cost_rejected = True
                warnings.warn(
                    "provider-reported cost is not finite and non-negative; "
                    "falling back to the price-table estimate",
                    RuntimeWarning,
                    stacklevel=2,
                )
        total = reported_cost if reported_cost is not None else estimated_total
        cost_source = (
            "provider_reported"
            if reported_cost is not None
            else (
                "price_table_invalid_provider_cost"
                if provider_cost_rejected
                else "price_table"
            )
        )

        with _LOCK, _ledger_file_lock(self.ledger_path):
            self.cumulative = self._read_cumulative()
            self.cumulative += total
            rec = CostRecord(
                timestamp=datetime.now(UTC).isoformat(),
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_usd=round(in_usd, 6),
                output_usd=round(out_usd, 6),
                total_usd=round(total, 6),
                cumulative_usd=round(self.cumulative, 6),
                purpose=purpose,
                reasoning_tokens=reasoning_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_write_tokens=cache_write_tokens,
                latency_seconds=(
                    round(latency_seconds, 6) if latency_seconds is not None else None
                ),
                reasoning_mode=reasoning_mode,
                reasoning_effort=reasoning_effort,
                estimated_total_usd=round(estimated_total, 6),
                provider_reported_cost_usd=(
                    round(reported_cost, 6) if reported_cost is not None else None
                ),
                provider_cost_rejected=provider_cost_rejected,
                cost_source=cost_source,
                transport_attempts=transport_attempts,
            )
            self._append(rec)

        self._enforce_caps(rec)
        return rec

    def assert_can_spend(self, planned_usd: float) -> None:
        """Pre-flight check before initiating an expensive batch."""
        with _LOCK, _ledger_file_lock(self.ledger_path):
            self.cumulative = self._read_cumulative()
        cap = self._limit("total_hard_cap")
        if cap is not None and self.cumulative + planned_usd > cap:
            raise BudgetExceeded(
                f"planned spend {planned_usd:.4f} would push cumulative "
                f"{self.cumulative:.4f} over hard cap {cap:.4f}"
            )
        # Per-run cap: abort when the spend of this run (cumulative - baseline)
        # exceeds ``run_hard_cap``; a parent process passes it through the
        # T_GADE_RUN_COST_CAP env var.
        run_cap = self._limit("run_hard_cap")
        if run_cap is not None:
            run_spend_now = max(0.0, self.cumulative - self.run_baseline_usd)
            if run_spend_now + planned_usd > run_cap:
                raise BudgetExceeded(
                    f"planned spend {planned_usd:.4f} would push run-cost "
                    f"{run_spend_now:.4f} over run_hard_cap {run_cap:.4f} "
                    f"(baseline={self.run_baseline_usd:.4f}, cumulative="
                    f"{self.cumulative:.4f})"
                )

    def run_cost(self) -> float:
        """Return USD spent in the current per-run window.

        ``run_baseline_usd`` is captured at tracker init from the
        ``T_GADE_RUN_BASELINE_USD`` env var if set, else from the cumulative
        at construction time.  ``run_cost = cumulative - baseline`` (clamped
        to ≥ 0).  Useful for run summaries and the ``run_hard_cap`` check.
        """
        return max(0.0, self.cumulative - self.run_baseline_usd)

    # ---------- internal ----------

    def _limit(self, key: str) -> float | None:
        limits = self.budgets.get("limits") or {}
        if key == "total_hard_cap":
            total_override = os.environ.get(_TOTAL_CAP_ENV)
            if total_override is not None:
                try:
                    value = float(total_override)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid {_TOTAL_CAP_ENV}: {total_override!r}"
                    ) from exc
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"invalid {_TOTAL_CAP_ENV}: {total_override!r}")
                expected_marker = f"runner_scoped_total_cap:{value:.6f}"
                if os.environ.get(_TOTAL_CAP_OPT_IN_ENV) != expected_marker:
                    raise ValueError(
                        f"{_TOTAL_CAP_ENV} requires matching {_TOTAL_CAP_OPT_IN_ENV}"
                    )
                return value
        if key == "run_hard_cap":
            run_override = os.environ.get("T_GADE_RUN_COST_CAP")
            if run_override is not None:
                return float(run_override)
        v = limits.get(key)
        if v is None:
            env_override = os.environ.get(f"GRANT_EVO_COST_{key.upper()}")
            if env_override:
                v = float(env_override)
        return float(v) if v is not None else None

    def _enforce_caps(self, rec: CostRecord) -> None:
        hard = self._limit("total_hard_cap")
        if hard is not None and rec.cumulative_usd > hard:
            raise BudgetExceeded(
                f"cumulative cost {rec.cumulative_usd:.4f} USD exceeded hard cap {hard:.4f}"
            )
        # Per-run cap.
        run_cap = self._limit("run_hard_cap")
        if run_cap is not None:
            run_spend = max(0.0, rec.cumulative_usd - self.run_baseline_usd)
            if run_spend > run_cap:
                raise BudgetExceeded(
                    f"run cost {run_spend:.4f} USD (cumulative {rec.cumulative_usd:.4f} "
                    f"− baseline {self.run_baseline_usd:.4f}) exceeded run_hard_cap "
                    f"{run_cap:.4f}"
                )
        per_call = self._limit("per_call_max")
        if per_call is not None and rec.total_usd > per_call:
            raise BudgetExceeded(
                f"single call cost {rec.total_usd:.4f} USD exceeded per_call_max {per_call:.4f}"
            )

    def _append(self, rec: CostRecord) -> None:
        line = json.dumps(rec.__dict__, ensure_ascii=False)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _read_cumulative(self) -> float:
        if not self.ledger_path.exists():
            return 0.0
        first_base: float | None = None
        total_sum = 0.0
        last_cumulative = 0.0
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        cumulative = float(rec.get("cumulative_usd", 0.0))
                        total = float(rec.get("total_usd", 0.0))
                        if first_base is None:
                            first_base = max(0.0, cumulative - total)
                        total_sum += total
                        last_cumulative = cumulative
                    except (KeyError, ValueError, json.JSONDecodeError):
                        pass
        reconstructed = (first_base or 0.0) + total_sum
        return max(reconstructed, last_cumulative)


@contextmanager
def _ledger_file_lock(ledger_path: Path):
    """Interprocess lock for ledger read-modify-append.

    The in-process thread lock above is insufficient when multiple pipeline
    subprocesses append to the same ledger.  This lock serializes the critical
    section without adding an external dependency.
    """
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"\0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_run_baseline_env(default_cumulative: float) -> float:
    """Read ``T_GADE_RUN_BASELINE_USD`` env var.

    Set by ``run_pipeline.py`` at the start of a multi-subprocess run so
    every subprocess shares the same notion of "this run started at USD X".
    Without the env var (e.g. running a single script directly), the
    baseline defaults to the cumulative at tracker construction time —
    effectively "everything spent before now is prior history".

    A malformed value is tolerated and treated as absence (no per-run
    accounting); we err on the side of not crashing the tracker for an
    env-var typo.
    """
    raw = os.environ.get("T_GADE_RUN_BASELINE_USD")
    if raw is None:
        return default_cumulative
    try:
        return float(raw)
    except ValueError:
        return default_cumulative


# ---------- module-level helper ----------

_tracker: CostTracker | None = None


def get_tracker(
    repo_root: Path | None = None,
) -> CostTracker:
    global _tracker
    if _tracker is None:
        if repo_root is None:
            # heuristic: assume cwd or parent contains config/
            here = Path.cwd()
            for candidate in [here, *here.parents]:
                if (candidate / "config" / "budgets.yaml").exists():
                    repo_root = candidate
                    break
            if repo_root is None:
                raise RuntimeError("cannot locate repo root with config/budgets.yaml")
        _tracker = CostTracker(
            budgets_path=repo_root / "config" / "budgets.yaml",
            prices_path=repo_root / "config" / "prices.yaml",
            ledger_path=repo_root / "outputs" / "cost_ledger.jsonl",
        )
    return _tracker


def reset_tracker_for_tests() -> None:
    global _tracker
    _tracker = None
