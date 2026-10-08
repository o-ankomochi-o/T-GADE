"""Produce and seal a shared gen0 bank for a task (the same producer as the
E4 campaign: MultistartRun with fresh_calls = 3N, stop at N valid i1 samples).

  python scripts/make_gen0_bank.py --task tsp_construct --seed 101 --confirm-paid --out data/gen0_bank_tsp.json

Paid (N..3N i1 calls of the prereg model); the call ledger stays in
<out>.producer/. The bank file is sealed with <out>.sha256 and its identity is
sha256 of the sorted genotype digests (bank_digest), as for data/gen0_bank.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from grant_evo.bench.e4_params import PREREG_PARAMS  # noqa: E402
from grant_evo.bench.multistart import MultistartRun, bank_from_result  # noqa: E402
from grant_evo.bench.seal import write_seal  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="tsp_construct", choices=["tsp_construct", "tsp_ga_crossover", "tsp_ga_suite"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ceiling", type=int, default=None, help="hard ceiling of i1 calls (default 3n, as in the E4 campaign)")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--client-seed", type=int, default=None, help="default 30000+seed")
    ap.add_argument("--cap-usd", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=None, help="override the prereg max_tokens (must match the arms)")
    ap.add_argument("--confirm-paid", action="store_true")
    ap.add_argument("--out", default=str(REPO / "data" / "gen0_bank_tsp.json"))
    a = ap.parse_args(argv)
    if not a.confirm_paid:
        print("REFUSED: paid producer needs --confirm-paid")
        return 2
    from grant_evo.bench.adapters.tsp_construct import TspConstructAdapter  # noqa: PLC0415
    from grant_evo.bench.clients import OpenRouterClient  # noqa: PLC0415
    if a.task == "tsp_ga_suite":
        from grant_evo.bench.adapters.tsp_ga_suite import TspGaSuiteAdapter  # noqa: PLC0415
        adapter = TspGaSuiteAdapter()
    elif a.task == "tsp_ga_crossover":
        from grant_evo.bench.adapters.tsp_ga_crossover import TspGaCrossoverAdapter  # noqa: PLC0415
        adapter = TspGaCrossoverAdapter()
    else:
        adapter = TspConstructAdapter()
    P = PREREG_PARAMS
    cseed = a.client_seed if a.client_seed is not None else 30000 + a.seed
    client = OpenRouterClient(P["model"], temperature=P["t_sample"], budget_usd=a.cap_usd,
                              retries=P["transport_retries"], min_interval_s=P["min_interval_s"],
                              max_tokens=(a.max_tokens or P["max_tokens"]), price_in_usd_per_m=P["price_in_usd_per_m"],
                              price_out_usd_per_m=P["price_out_usd_per_m"], max_input_bytes=P["max_input_bytes"],
                              chat_overhead_tokens=P["chat_overhead_tokens"], provider_order=P["provider_order"],
                              allow_fallbacks=P["allow_fallbacks"], seed=cseed)
    out = Path(a.out)
    pdir = out.with_name(out.name + ".producer_" + time.strftime("%Y%m%dT%H%M%S"))
    prod = MultistartRun(adapter, client, fresh_calls=a.ceiling or 3 * a.n, stop_at_valid=a.n, out_dir=str(pdir),
                         seed=a.seed, label="bank").run()
    valid = [(c["genes"], c) for c in prod["candidates"] if c["origin"] == "fresh" and c["energy"] is not None]
    print(json.dumps({"attempts": prod["attempts"], "valid": len(valid), "spent_usd": getattr(client, "spent_usd", None),
                      "energies": [round(c["energy"], 4) for _g, c in valid]}))
    if len(valid) < a.n:
        print("BANK_SHORT: fewer valid genotypes than n; nothing sealed")
        return 1
    bank = bank_from_result(prod, [g for g, _c in valid[:a.n]])
    bank["task"] = a.task
    bank["model"] = P["model"]
    bank["client_seed"] = cseed
    body = json.dumps(bank, indent=1, sort_keys=True)
    out.write_text(body, encoding="utf-8", newline="\n")
    sha = write_seal(out)
    print("sealed", out, "file sha256", sha, "bank sha256", bank["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
