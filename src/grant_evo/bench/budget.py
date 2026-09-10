"""Global cross-run budget authority for the E4 campaign.

The ledger records a conservative reservation *before* transport.  A normal
return settles that row to the actual charge; a crash leaves the full
reservation committed.  E4 arms are deliberately sequential, so an atomic
replace is sufficient here (parallel arms require a real inter-process lock).
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path


class GlobalBudget:
    def __init__(self, path: "str | Path", cap_usd: float):
        self.path = Path(path)
        self.cap_usd = self._amount("cap_usd", cap_usd)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._store({"cap_usd": self.cap_usd, "rows": []})
        recorded = float(self._load()["cap_usd"])
        if recorded != self.cap_usd:
            raise ValueError(
                f"budget cap mismatch: ledger={recorded}, requested={self.cap_usd}")

    @staticmethod
    def _amount(name: str, value: float) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")
        return float(value)

    def _load(self) -> dict:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise ValueError("invalid global-budget ledger schema")
        self._amount("ledger cap_usd", data.get("cap_usd"))
        for row in data["rows"]:
            if not isinstance(row, dict):
                raise ValueError("invalid global-budget row")
            self._amount("ledger row usd", row.get("usd"))
        return data

    def _store(self, data: dict) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(data, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def committed_usd(self) -> float:
        return float(sum(r["usd"] for r in self._load()["rows"]))

    def check(self, reserve_usd: float) -> None:
        reserve_usd = self._amount("reserve_usd", reserve_usd)
        total = self.committed_usd()
        if total + reserve_usd > self.cap_usd:
            from grant_evo.bench.clients import BudgetExceeded  # noqa: PLC0415
            raise BudgetExceeded(
                f"GLOBAL cap {self.cap_usd} USD: committed {total:.4f} + "
                f"reserve {reserve_usd:.6f} would exceed it")

    def reserve(self, usd: float, note: str = "") -> str:
        """Durably reserve a worst-case charge before transport."""
        usd = self._amount("reservation usd", usd)
        data = self._load()
        total = float(sum(float(row["usd"]) for row in data["rows"]))
        if total + usd > self.cap_usd:
            from grant_evo.bench.clients import BudgetExceeded  # noqa: PLC0415
            raise BudgetExceeded(
                f"GLOBAL cap {self.cap_usd} USD: committed {total:.4f} + "
                f"reserve {usd:.6f} would exceed it")
        reservation_id = uuid.uuid4().hex
        data["rows"].append({"id": reservation_id, "usd": usd,
                             "status": "reserved", "note": note,
                             "ts": time.time()})
        self._store(data)
        return reservation_id

    def settle(self, reservation_id: str, usd: float, note: str = "") -> None:
        """Replace one reservation with the charge actually incurred."""
        usd = self._amount("settled usd", usd)
        data = self._load()
        matches = [row for row in data["rows"]
                   if row.get("id") == reservation_id]
        if len(matches) != 1 or matches[0].get("status") != "reserved":
            raise ValueError("unknown or already-settled budget reservation")
        row = matches[0]
        row.update({"usd": usd, "status": "charged", "note": note,
                    "settled_ts": time.time()})
        self._store(data)

    def charge(self, usd: float, note: str = "") -> None:
        """Compatibility path for non-transport charges."""
        usd = self._amount("charge usd", usd)
        self.check(usd)
        d = self._load()
        d["rows"].append({"id": uuid.uuid4().hex, "usd": usd,
                          "status": "charged", "note": note,
                          "ts": time.time()})
        self._store(d)
