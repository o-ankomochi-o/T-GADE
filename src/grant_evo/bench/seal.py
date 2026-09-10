"""ONE seal convention for every sealed artifact ( the
freezer and the runner disagreed on the seal file name).

seal_path(p) == p.name + ".sha256" next to p (e.g. freeze.json.sha256).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def seal_path(path: "str | Path") -> Path:
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def write_seal(path: "str | Path", sha: "str | None" = None) -> str:
    p = Path(path)
    sha = sha or sha256_bytes(p.read_bytes())
    seal_path(p).write_text(sha + "\n", encoding="utf-8", newline="\n")
    return sha


def check_seal(path: "str | Path") -> bool:
    p = Path(path)
    s = seal_path(p)
    return p.exists() and s.exists() and s.read_text(encoding="utf-8").strip() == sha256_bytes(p.read_bytes())


def read_seal(path: "str | Path") -> "str | None":
    s = seal_path(path)
    return s.read_text(encoding="utf-8").strip() if s.exists() else None
