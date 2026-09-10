"""Least-privilege Docker sandbox for GENERATED candidate code.

Two boundaries, not one:

1. HOST boundary (T60 P0-B): every execution is a disposable container —
   --network none, --read-only root (tmpfs /tmp), the vendored EoH repo as
   the ONLY mount (read-only), --cap-drop ALL (+SETUID/SETGID only, used to
   demote the child), --pids-limit, memory/CPU capped.

2. MEASUREMENT boundary (T90 counterexample O): candidate code runs in a
   SEPARATE, UID-DEMOTED child process (nobody) inside the container and is
   used ONLY as a score oracle over a seq-tagged pipe protocol. The trusted
   supervisor (container root) owns the official evaluation math and emits
   the ONLY protocol record on its own stdout — the candidate has no handle
   to it (different process AND different uid, so /proc/<sup>/fd is closed).
   Child stderr is discarded; child stdout junk is tolerated up to a line
   budget and never trusted except as {"seq": n, "scores": [...]} answers,
   whose content is the candidate's legitimate output domain anyway.
   Unexpected child exit, malformed protocol, or timeout => invalid (None).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

IMAGE = "tgade-sandbox:1"
_EOH_DIR = Path(__file__).resolve().parents[3] / "third_party" / "EoH"
# Evaluator protocol version: recorded by runners next to the
# image digest. "json-line-v0" = HEAD b0ef2e3 (E4 pilot); "binary-f64-v1" =
# length-prefixed float64 pipes with candidate fd 0/1 on /dev/null (V101).
PROTOCOL = "binary-f64-v1"
_image_digest_cache: "str | None" = None

# Inner child: exec the candidate ONCE, then serve score queries. Runs as
# nobody. Its stdout is a protocol pipe to the supervisor, never the record.
_CHILD = r'''
import json, os, struct, sys
import numpy as np
# Binary protocol (V101 P1): the JSON-line protocol cost 1.99 ms per score()
# call (25000 calls = 50 s per evaluation); length-prefixed float64 costs
# 0.12 ms. The protocol pipes are PRIVATE duplicates of fd 0/1 taken before
# the candidate runs; fd 0/1 (and sys.stdin/stdout) are then /dev/null, so
# candidate prints/reads can neither corrupt nor consume the stream.
_in = os.fdopen(os.dup(0), "rb")
_out = os.fdopen(os.dup(1), "wb")
_null = os.open(os.devnull, os.O_RDWR)
os.dup2(_null, 0)
os.dup2(_null, 1)
sys.stdin = open(os.devnull, "r")
sys.stdout = open(os.devnull, "w")
spec = json.loads(_in.readline())
if spec.get("eval_seed") is not None:
    # evaluation-seed panel: fixed RNG state for candidate code that draws random numbers
    # (legacy np.random.*, random.*, and the Generator API np.random.default_rng() with no seed;
    # counterexample 2026-09-07)
    import random as _rnd
    _es = int(spec["eval_seed"])
    _rnd.seed(_es)
    np.random.seed(_es % (2 ** 32))
    _orig_default_rng = np.random.default_rng
    _rng_calls = [0]
    def _seeded_default_rng(seed=None, *a, **k):
        # each unseeded generator gets a distinct but deterministic seed ( two
        # unseeded generators must not produce identical streams)
        if seed is None:
            _rng_calls[0] += 1
            seed = (_es * 1000003 + _rng_calls[0]) % (2 ** 32)
        return _orig_default_rng(seed, *a, **k)
    np.random.default_rng = _seeded_default_rng
ns = {"np": np}
exec(spec["code"], ns)
score = ns["score"]
_out.write(struct.pack("<ii", 0, 0))  # ready marker
_out.flush()

def _read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)

while True:
    head = _read_exact(16)
    if head is None:
        break
    seq, n, item = struct.unpack("<iiq", head)  # item is an int, as before
    raw = _read_exact(8 * n)
    if raw is None:
        break
    try:
        s = score(item, np.frombuffer(raw, dtype="<f8").copy())
        arr = np.asarray(s, dtype=float).ravel()
        payload, m = arr.astype("<f8").tobytes(), arr.shape[0]
    except Exception:
        payload, m = b"", -1  # supervisor treats m != n as Invalid
    _out.write(struct.pack("<ii", seq, m) + payload)
    _out.flush()
'''

# Trusted supervisor: official evaluation math; candidate = score oracle.
_HARNESS = r'''
import json, os, struct, subprocess, sys
import numpy as np

CHILD_SRC = __CHILD_SRC__

def demote():
    os.setgid(65534)
    os.setuid(65534)

class Invalid(Exception):
    pass

class Oracle:
    """Seq-tagged score oracle over the demoted child's pipes. Binary,
    length-prefixed float64 (V101 P1); ANY protocol deviation (wrong seq,
    wrong length, EOF) is Invalid -- there is no junk tolerance because the
    child's fd 0/1 are /dev/null for the candidate (see _CHILD)."""

    def __init__(self, code, eval_seed=None):
        self.p = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, preexec_fn=demote)
        self.seq = 0
        self.p.stdin.write((json.dumps({"code": code, "eval_seed": eval_seed}) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None or struct.unpack("<ii", head) != (0, 0):
            raise Invalid("child failed to initialise candidate")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:  # EOF: child died (os._exit etc.)
                return None
            buf += chunk
        return bytes(buf)

    def __call__(self, item, bins):
        self.seq += 1
        n = self.seq
        b = np.ascontiguousarray(bins, dtype="<f8").ravel()
        self.p.stdin.write(struct.pack("<iiq", n, b.shape[0], int(item)) + b.tobytes())
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None:
            raise Invalid("candidate score call failed")
        seq, m = struct.unpack("<ii", head)
        if seq != n or m != b.shape[0]:
            raise Invalid("candidate score call failed")
        raw = self._read_exact(8 * m)
        if raw is None:
            raise Invalid("candidate score call failed")
        return np.frombuffer(raw, dtype="<f8").astype(float)

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass

req = json.loads(sys.stdin.read())
sys.path.insert(0, "/eoh/examples/bp_online")
sys.path.insert(0, "/eoh/eoh/src")
result = {"ok": False, "error": "unknown op"}
oracle = None
try:
    oracle = Oracle(req["code"], req.get("eval_seed"))
    if req["op"] == "energy":
        import types
        if "requests" not in sys.modules:  # eoh imports it; transport unused
            stub = types.ModuleType("requests")
            def _na(*a, **k):
                raise RuntimeError("requests stub")
            stub.post = _na
            stub.get = _na
            sys.modules["requests"] = stub
        from prob import BPONLINE
        pb = BPONLINE(capacity=req["capacity"], timeout=10**9)
        pb.instances, pb.lb = req["instances"], req["lb"]
        # OFFICIAL measurement math (evaluate_program/online_binpack) runs
        # HERE in the supervisor; only score() is delegated to the child.
        rec = []
        n0 = 0
        if req.get("signature"):
            # By-product behaviour signature: (fill after placement, tightness rank)
            # of EVERY step of the FIRST instance, taken from the official loop's own oracle calls.
            # Same formulas as the "signature" op; no extra execution.
            n0 = len(next(iter(next(iter(req["instances"].values())).values()))["items"])
            cap = float(req["capacity"])
            def oracle_rec(item, bins):
                sc = oracle(item, bins)
                if len(rec) < 2 * n0:
                    b = np.asarray(bins, dtype=float)
                    k = int(np.argmax(sc))
                    order = np.argsort(b, kind="stable")
                    rank = int(np.where(order == k)[0][0]) / max(1, len(b) - 1)
                    rec.append(round(float(1.0 - (b[k] - float(item)) / cap), 4))
                    rec.append(round(float(rank), 4))
                return sc
        out = pb.evaluate_program("", oracle_rec if n0 else oracle)
        result = {"ok": True, "value": out, **({"signature": rec} if n0 else {})}
    elif req["op"] == "signature":
        capacity = req["capacity"]
        bins = np.array([float(capacity)] * len(req["items"]))
        feat = []
        for it in req["items"]:
            valid = np.nonzero((bins - it) >= 0)[0]
            sc = oracle(it, bins[valid])
            k = int(np.argmax(sc))
            b = valid[k]
            order = np.argsort(bins[valid], kind="stable")
            rank = int(np.where(order == k)[0][0]) / max(1, len(valid) - 1)
            bins[b] -= it
            feat.append(round(float(1.0 - bins[b] / capacity), 4))
            feat.append(round(float(rank), 4))
        result = {"ok": True, "value": feat}
except Invalid as exc:
    result = {"ok": False, "error": f"Invalid: {exc}"}
except Exception as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
finally:
    if oracle is not None:
        oracle.close()
# the ONLY protocol record; candidate has no path to this stream.
print(json.dumps(result))
'''.replace("__CHILD_SRC__", json.dumps(_CHILD))


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_image() -> None:
    """Build the pinned sandbox image if absent (network used at BUILD only)."""
    have = subprocess.run(["docker", "image", "inspect", IMAGE],
                          capture_output=True)
    if have.returncode == 0:
        return
    dockerfile = "FROM python:3.11-slim\nRUN pip install --no-cache-dir numpy==2.4.4\n"
    r = subprocess.run(["docker", "build", "-t", IMAGE, "-"],
                       input=dockerfile.encode("utf-8"), capture_output=True,
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"sandbox image build failed: {r.stderr.decode()[-400:]}")


def image_digest() -> "str | None":
    """Docker image ID of the sandbox (recorded in run reports;)."""
    global _image_digest_cache
    if _image_digest_cache is None:
        r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                          capture_output=True, text=True)
        if r.returncode == 0:
            _image_digest_cache = r.stdout.strip()
    return _image_digest_cache


def run_sandboxed(payload: dict, timeout: int) -> "dict | None":
    """One candidate execution. Returns the supervisor JSON or None
    (fail-closed on timeout, abnormal exit, or malformed record)."""
    name = f"tgade_sb_{uuid.uuid4().hex[:12]}"
    cmd = ["docker", "run", "--rm", "-i", "--name", name,
           "--network", "none", "--read-only", "--tmpfs", "/tmp",
           "--memory", "1g", "--cpus", "1", "--pids-limit", "128",
           "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
           "--security-opt", "no-new-privileges",
           "-v", f"{_EOH_DIR}:/eoh:ro",
           IMAGE, "python", "-I", "-c", _HARNESS]
    try:
        r = subprocess.run(cmd, input=json.dumps(payload), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.strip().splitlines()
    if len(lines) != 1:
        # the supervisor emits EXACTLY one record; anything else is invalid
        # (defense in depth — the candidate cannot reach this stream anyway).
        return None
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError:
        return None
