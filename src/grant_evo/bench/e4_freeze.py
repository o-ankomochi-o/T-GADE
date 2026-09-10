"""E4 freeze artifact: creation and verification share ONE implementation
. Computer-science machinery only.

The artifact fixes: the parameter-table blob (== git HEAD blob, path clean), the E3
report digest/path/source digest, the E3 commit (== the code commit: E3
evidence must be measured at the frozen commit), the code commit (bench
paths clean), and the freeze message id. verify_freeze() re-checks
every one of these and re-runs the E3 gate.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from grant_evo.bench.e3_gate import validate_e3_report
from grant_evo.bench.seal import check_seal, sha256_bytes, write_seal

REPO = Path(__file__).resolve().parents[3]
PREREG_REL = "experiments/e4/prereg_bp_online.md"
BENCH_PATHS = ["src", "scripts/run_e4_campaign.py", "scripts/run_e3_smoke.py",
               "scripts/freeze_e4.py", "rust/tgade_core/src", PREREG_REL,
               "experiments/e4/bank_seal.sha256"]


class FreezeError(RuntimeError):
    pass


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True,
                          text=True, check=True).stdout.strip()


def head_commit() -> str:
    return _git("rev-parse", "HEAD")


def head_prereg_blob() -> str:
    return _git("rev-parse", f"HEAD:{PREREG_REL}")


def bench_dirty() -> str:
    """git status --porcelain over the bench paths (untracked included)."""
    return _git("status", "--porcelain", "--", *BENCH_PATHS)


def create_freeze(e3_report: "str | Path", freeze_message_id: str, out: "str | Path",
                  *, require_clean: bool = True) -> dict:
    dirt = bench_dirty()
    if require_clean and dirt:
        raise FreezeError("bench paths are not clean:\n" + dirt)
    if not str(freeze_message_id).strip():
        raise FreezeError("a freeze message id is required")
    rep_path = Path(e3_report)
    commit = head_commit()
    # the E3 evidence must have been measured at THIS commit.
    rep = validate_e3_report(rep_path, expected_commit=commit, require_clean=require_clean)
    art = {
        "prereg_blob_sha": head_prereg_blob(),
        "prereg_path": PREREG_REL,
        "e3_report_path": str(rep_path),
        "e3_report_sha256": sha256_bytes(rep_path.read_bytes()),
        "e3_source_result_sha256": (rep.get("source_run") or {}).get("result_sha256"),
        "e3_commit": rep["git_commit"],
        "code_commit": commit,
        "freeze_message_id": str(freeze_message_id),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(art, indent=1, sort_keys=True)
    outp.write_text(body, encoding="utf-8", newline="\n")
    write_seal(outp, sha256_bytes(body.encode("utf-8")))
    return art


def verify_freeze(freeze_path: "str | Path", *, require_clean: bool = True) -> dict:
    fp = Path(freeze_path)
    if not check_seal(fp):
        raise FreezeError("freeze artifact seal missing or mismatched")
    art = json.loads(fp.read_bytes())
    if art.get("prereg_blob_sha") != head_prereg_blob():
        raise FreezeError("parameter-table blob at HEAD differs from the frozen blob")
    dirt = bench_dirty()
    if require_clean and dirt:
        raise FreezeError("bench paths are not clean (working tree differs from HEAD)")
    commit = head_commit()
    if art.get("code_commit") != commit:
        raise FreezeError("code commit differs from the frozen commit")
    if art.get("e3_commit") != art.get("code_commit"):
        raise FreezeError("E3 evidence was not measured at the frozen code commit")
    rep_path = Path(art["e3_report_path"])
    if not rep_path.is_absolute():
        rep_path = REPO / rep_path
    rep = validate_e3_report(rep_path, expected_sha=art["e3_report_sha256"],
                             expected_commit=art["e3_commit"], require_clean=require_clean)
    if (rep.get("source_run") or {}).get("result_sha256") != art.get("e3_source_result_sha256"):
        raise FreezeError("E3 source result digest differs from the frozen value")
    if not str(art.get("freeze_message_id", "")).strip():
        raise FreezeError("freeze artifact carries no freeze message id")
    art["_t_star"] = rep["t_star_e4_rule"]
    return art
