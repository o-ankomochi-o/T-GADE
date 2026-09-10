"""Append structured lineage events to ``outputs/lineage.jsonl``.

Every LLM call, every new individual, every user decision, every constraint
check produces one JSONL line. The file is append-only and is the audit trail
that lets a different person (e.g. another applicant trying the same engine)
understand what the system did and why.

Schema (one JSON object per line)::

    {
      "timestamp": "2026-04-25T03:14:15Z",
      "kind": "call" | "individual" | "decision" | "constraint" | "phase",
      "session_id": "...",
      "id": "...",        # event-specific id
      "context": {...},   # event-specific payload
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


class LineageLogger:
    """Append-only JSONL logger."""

    def __init__(self, path: Path, session_id: str | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or _short_uuid()

    def log(self, kind: str, event_id: str | None = None, **context: Any) -> str:
        eid = event_id or _short_uuid()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "session_id": self.session_id,
            "id": eid,
            "context": context,
        }
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return eid


def text_digest(text: str, length: int = 12) -> str:
    """Short content-addressable digest of a prompt or response."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ---------- module-level helper ----------

_logger: LineageLogger | None = None


def get_logger(
    repo_root: Path | None = None,
    session_id: str | None = None,
) -> LineageLogger:
    global _logger
    if _logger is None:
        if repo_root is None:
            here = Path.cwd()
            for candidate in [here, *here.parents]:
                if (candidate / "config").is_dir() and (candidate / "src").is_dir():
                    repo_root = candidate
                    break
            if repo_root is None:
                repo_root = Path.cwd()
        _logger = LineageLogger(
            path=repo_root / "outputs" / "lineage.jsonl",
            session_id=session_id or os.environ.get("GRANT_EVO_SESSION_ID"),
        )
    return _logger


def reset_logger_for_tests() -> None:
    global _logger
    _logger = None
