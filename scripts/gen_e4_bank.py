"""Generate, seal, and VERIFY the E4 confirmation banks ($0, offline).

Generator family matched to the vendored Weibull-5k training bank
(mean 40.24, min 1, max 92 over 5000 items):
items = clip(round(Weibull(shape=3, scale=45)), 1, 100), integers, PCG64.

Banks (both evaluation-only; they NEVER enter prompts, selection,
calibration, or tuning):
  primary   capacity 100, seeds 90001-90005 -> confirmation_bank.json
  secondary capacity 500, seeds 90101-90105 -> confirmation_bank_c500.json


`--verify` regenerates every sequence from the declared generator and
checks each file's bytes against experiments/e4/bank_seal.sha256
(reproducible generator/verifier for the c500 bank).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
E4 = REPO / "experiments" / "e4"
SEAL = E4 / "bank_seal.sha256"
BANKS = {
    "confirmation_bank.json": {"capacity": 100, "seeds": [90001, 90002, 90003, 90004, 90005],
                               "prefix": "conf"},
    "confirmation_bank_c500.json": {"capacity": 500,
                                    "seeds": [90101, 90102, 90103, 90104, 90105],
                                    "prefix": "conf500"},
}
N_ITEMS = 5000


def _items(seed: int) -> list[int]:
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.clip(np.round(rng.weibull(3.0, N_ITEMS) * 45.0), 1, 100).astype(int).tolist()


def _payload(spec: dict) -> bytes:
    inst = {f"{spec['prefix']}_{seed}": {"capacity": spec["capacity"],
                                         "num_items": N_ITEMS, "seed": seed,
                                         "items": _items(seed)}
            for seed in spec["seeds"]}
    return json.dumps({"generator": "clip(round(Weibull(k=3, lambda=45)), 1, 100)",
                       "seeds": spec["seeds"], "capacity": spec["capacity"],
                       "instances": inst}, sort_keys=True).encode("utf-8")


def _sealed() -> dict:
    if not SEAL.exists():
        return {}
    return dict(line.split(" sha256 ")
                for line in SEAL.read_text(encoding="utf-8").splitlines() if line)


def generate() -> int:
    sealed = _sealed()
    for name, spec in BANKS.items():
        path = E4 / name
        if path.exists() or name in sealed:
            print(f"SKIP {name}: already generated/sealed (frozen once, never regenerated)")
            continue
        blob = _payload(spec)
        path.write_bytes(blob)
        sha = hashlib.sha256(blob).hexdigest()
        with open(SEAL, "ab") as fh:
            fh.write(f"{name} sha256 {sha}\n".encode("utf-8"))
        print(json.dumps({"bank": name, "sha256": sha, "instances": len(spec["seeds"])}))
    if "vendored_training_bank" not in sealed:
        sys.path.insert(0, str(REPO / "third_party/EoH/examples/bp_online"))
        sys.path.insert(0, str(REPO / "third_party/EoH/eoh/src"))
        import types
        sys.modules.setdefault("requests", types.ModuleType("requests"))
        from get_instance import GetData
        train, _ = GetData().get_instances(100)
        train_sha = hashlib.sha256(json.dumps(
            train, sort_keys=True, default=int).encode("utf-8")).hexdigest()
        with open(SEAL, "ab") as fh:
            fh.write(f"vendored_training_bank sha256 {train_sha}\n".encode("utf-8"))
        print(json.dumps({"vendored_training_bank": train_sha}))
    return 0


def verify() -> int:
    sealed = _sealed()
    ok = True
    for name, spec in BANKS.items():
        path = E4 / name
        if name not in sealed or not path.exists():
            print(f"FAIL {name}: missing file or seal")
            ok = False
            continue
        on_disk = hashlib.sha256(path.read_bytes()).hexdigest()
        regenerated = hashlib.sha256(_payload(spec)).hexdigest()
        status = "OK" if on_disk == sealed[name] == regenerated else "FAIL"
        ok &= status == "OK"
        print(f"{status} {name}: disk={on_disk[:12]} seal={sealed[name][:12]} "
              f"regen={regenerated[:12]}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    raise SystemExit(verify() if a.verify else generate())
