"""Single source of the FROZEN execution parameters.

Used by the paid campaign runner, the E3 smoke runner (so the E3 report
records the same execution conditions) and the E3 gate (which verifies the
report against this table). Computer-science machinery only.
"""

from __future__ import annotations

MODEL = "qwen/qwen3-32b"

# Provider pinning: one provider, no fallbacks. A
# provider failure is a transport-class failure -> bounded retry -> vacancy.
PROVIDER_ORDER = ["SiliconFlow"]
ALLOW_FALLBACKS = False

PREREG_PARAMS = {
    "model": MODEL,
    "n": 8, "generations": 10,
    "seeds": [101, 102, 103, 104, 105],
    "reserve_seeds": [106, 107, 108, 109, 110],
    "cap_usd": 10.0,
    "t_sample": 0.8, "strength": "mid", "occupancy": "boson",
    "max_tokens": 2048, "max_input_bytes": 60000, "chat_overhead_tokens": 512,
    "price_in_usd_per_m": 1.0, "price_out_usd_per_m": 2.0,
    "transport_retries": 2, "min_interval_s": 1.0,
    "provider_order": PROVIDER_ORDER, "allow_fallbacks": ALLOW_FALLBACKS,
}

# Execution conditions that MUST be identical between the E3 evidence and
# the E4 arms (verified by the gate against the E3 report's design block).
EXECUTION_KEYS = ("model", "max_tokens", "max_input_bytes", "chat_overhead_tokens",
                  "price_in_usd_per_m", "price_out_usd_per_m", "transport_retries",
                  "min_interval_s", "provider_order", "allow_fallbacks")


def execution_conditions() -> dict:
    return {k: PREREG_PARAMS[k] for k in EXECUTION_KEYS}
