"""Replay client for the S0 identity check (standalone only; no network, no key).

Serves the responses of a RECORDED run back to the EoH loop in call order: the recorded
``call_ledger_eoh.jsonl`` is consumed by logical call index (``call_id = eoh-<seq>``), the
prompt sha256 of each call must equal the recorded one, and the response text is read from
``responses/<response_sha256>.txt``. A recorded failure (ok=false) is replayed as a failure with
the recorded usage. Any deviation (unknown seq, prompt mismatch, missing response file, seq used
twice, call beyond the record) is written to ``replay_mismatch.jsonl`` in the replaying run's
output directory and raised as ``ReplayMismatch``; the shim re-raises it so the search stops
instead of continuing with an empty generation. The client never converts a mismatch into text.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from grant_evo.tgade.engine import LLMResult


class ReplayMismatch(RuntimeError):
    """Raised on the first deviation from the recorded call sequence (fail-closed)."""


class RecordedFailure(RuntimeError):
    """The recorded call failed (ok=false); replayed as a failure with the recorded usage."""

    def __init__(self, msg: str, usage: dict):
        super().__init__(msg)
        self.usage = usage


class ReplayClient:
    def __init__(self, record_dir: str | Path, out_dir: str | Path):
        self.record_dir = Path(record_dir)
        self.out_dir = Path(out_dir)
        self.started: dict[int, dict] = {}
        self.terminal: dict[int, dict] = {}
        ledger = self.record_dir / "call_ledger_eoh.jsonl"
        if not ledger.exists():
            raise FileNotFoundError(f"no recorded ledger: {ledger}")
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "call_started":
                self.started[int(row["seq"])] = row
            elif row.get("event") == "call_terminal":
                self.terminal[int(row["seq"])] = row
        self.expected = len(self.started)
        self.used: set[int] = set()
        self.mismatches = 0
        self.poisoned: str | None = None
        self._lock = threading.Lock()
        self._mm = open(self.out_dir / "replay_mismatch.jsonl", "a", encoding="utf-8", newline="\n")
        # hard stop by default (TGADE_REPLAY_SOFT=1 keeps the process alive for the self-check only)
        self.hard_stop = os.environ.get("TGADE_REPLAY_SOFT", "") != "1"
        self._closed = False
        atexit.register(self.close)
        self._log({"event": "replay_open", "ts": time.time(), "record_dir": str(self.record_dir),
                   "expected_calls": self.expected, "hard_stop": self.hard_stop})

    def _log(self, obj: dict) -> None:
        self._mm.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._mm.flush()

    def _fail(self, seq, kind: str, detail: dict) -> ReplayMismatch:
        with self._lock:
            self.mismatches += 1
            self.poisoned = self.poisoned or f"{kind} at seq {seq}"
            self._log({"event": "replay_mismatch", "ts": time.time(), "seq": seq, "kind": kind, **detail})
        if self.hard_stop:  # explicit stop path: the EoH loop swallows exceptions from get_response, so end the process
            self._log({"event": "replay_close", "ts": time.time(), **self.summary(), "hard_stop": True})
            self._mm.flush(); os.fsync(self._mm.fileno())
            sys.stderr.write(f"REPLAY MISMATCH ({kind}) at seq {seq}: {detail} -> hard stop (exit 3)\n"); sys.stderr.flush()
            os._exit(3)
        return ReplayMismatch(f"replay mismatch ({kind}) at seq {seq}: {detail}")

    def __call__(self, prompt: str, **opts) -> LLMResult:
        if self.poisoned:
            raise ReplayMismatch(f"replay poisoned: {self.poisoned}")
        call_id = opts.get("call_id")
        try:
            seq = int(str(call_id).split("-")[-1])
        except Exception:
            raise self._fail(None, "bad_call_id", {"call_id": call_id})
        with self._lock:
            if seq in self.used:
                raise self._fail(seq, "seq_reused", {})
            self.used.add(seq)
        rec = self.started.get(seq)
        if rec is None:
            raise self._fail(seq, "beyond_record", {"expected_calls": self.expected})
        psha = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
        if psha != rec.get("prompt_sha256"):
            raise self._fail(seq, "prompt_mismatch", {"recorded": rec.get("prompt_sha256"), "got": psha})
        term = self.terminal.get(seq)
        if term is None:
            raise self._fail(seq, "no_terminal_row", {})
        usage = dict(term.get("usage") or {})
        if not term.get("ok"):
            raise RecordedFailure(f"recorded failure at seq {seq}: {term.get('error')}", usage)
        rsha = term.get("response_sha256")
        f = self.record_dir / "responses" / f"{rsha}.txt"
        if not rsha or not f.exists():
            raise self._fail(seq, "missing_response_file", {"response_sha256": rsha})
        text = f.read_bytes().decode("utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != rsha:
            raise self._fail(seq, "response_hash_mismatch", {"response_sha256": rsha})
        return LLMResult(text=text, usage=usage)

    def summary(self) -> dict:
        return {"expected_calls": self.expected, "consumed": len(self.used), "mismatches": self.mismatches,
                "poisoned": self.poisoned}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._log({"event": "replay_close", "ts": time.time(), **self.summary()})
        self._mm.close()


if __name__ == "__main__":  # self-check without any run: a two-call synthetic record
    import tempfile, os
    t = Path(tempfile.mkdtemp()); (t / "responses").mkdir(); out = Path(tempfile.mkdtemp())
    p1, p2 = "hello", "world"; r1 = "resp-1"
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    (t / "responses" / f"{h(r1)}.txt").write_bytes(r1.encode())
    rows = [{"event": "call_started", "seq": 1, "prompt_sha256": h(p1)}, {"event": "call_terminal", "seq": 1, "ok": True, "response_sha256": h(r1), "usage": {"prompt_tokens": 1}},
            {"event": "call_started", "seq": 2, "prompt_sha256": h(p2)}, {"event": "call_terminal", "seq": 2, "ok": False, "error": "x", "usage": {}}]
    (t / "call_ledger_eoh.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    c = ReplayClient(t, out)
    assert c("hello", call_id="eoh-1").text == r1
    try:
        c("world", call_id="eoh-2"); raise AssertionError("recorded failure must raise")
    except RecordedFailure:
        pass
    c2 = ReplayClient(t, out)
    try:
        c2("HELLO", call_id="eoh-1"); raise AssertionError("prompt mismatch must raise")
    except ReplayMismatch:
        pass
    try:
        c2("hello", call_id="eoh-1"); raise AssertionError("poisoned client must keep raising")
    except ReplayMismatch:
        pass
    c3 = ReplayClient(t, out)
    try:
        c3("x", call_id="eoh-3"); raise AssertionError("beyond record must raise")
    except ReplayMismatch:
        pass
    c.close(); c2.close(); c3.close()
    print("replay_client selftest OK")
