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
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console pop-ups from console-less parents (2026-09-30)
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
    elif req["op"] == "energy_noisy":
        # NOISY prereg (2026-10-01): the policy sees an INTEGER observation x = clip(round(w + eps), 1, C),
        # eps ~ N(0, sigma^2); the valid mask and score() use x; the true w is applied after placement.
        # Overflow (w > rem) closes the bin (rem = -1) with the item left inside and counts O += 1.
        # Empty bins (rem == C) are preallocated as in the official loop, so choosing one opens a new bin.
        # J = B + kappa * O per instance; value = mean over datasets of (mean J - lb) / lb (official shape).
        # sigma = 0 reproduces the official loop exactly (x == w, no overflow possible).
        cap = float(req["capacity"]); sigma = float(req["sigma"]); kappa = float(req["kappa"])
        seed0 = int(req["noise_seed"]); n0 = 0; rec = []
        if req.get("signature"):
            n0 = len(next(iter(next(iter(req["instances"].values())).values()))["items"])
        per = {}; fit = []; k_inst = 0
        for dsname, ds in req["instances"].items():
            js = []
            for iname, inst in ds.items():
                rng = np.random.default_rng([seed0, k_inst]); k_inst += 1
                w = np.asarray(inst["items"], dtype=float)
                x = np.clip(np.rint(w + sigma * rng.standard_normal(w.shape[0])), 1, cap)
                bins = np.full(w.shape[0], cap); O = 0
                for t in range(w.shape[0]):
                    valid = np.nonzero(bins >= x[t])[0]
                    sc = oracle(int(x[t]), bins[valid])
                    k = int(np.argmax(sc)); b = int(valid[k])
                    if len(rec) < 2 * n0:
                        order = np.argsort(bins[valid], kind="stable")
                        rec.append(round(float(1.0 - (bins[b] - x[t]) / cap), 4))
                        rec.append(round(float(int(np.where(order == k)[0][0]) / max(1, len(valid) - 1)), 4))
                    if w[t] <= bins[b]:
                        bins[b] -= w[t]
                    else:
                        O += 1; bins[b] = -1.0
                B = int(np.sum(bins != cap)); J = B + kappa * O
                per[dsname + "/" + str(iname)] = [B, O, J]; js.append(J)
            lb = float(req["lb"][dsname])
            fit.append((float(np.mean(js)) - lb) / lb)
        result = {"ok": True, "value": float(np.mean(fit)), "per_instance": per, **({"signature": rec} if n0 else {})}
    elif req["op"] == "signature":
        capacity = req["capacity"]
        bins = np.array([float(capacity)] * len(req["items"]))
        feat = []
        # choices: steps whose feasible bins differ in remaining capacity (equal-capacity bins are interchangeable);
        # ties: those of them where bins of different capacity share the top score (argmax tie-break decides).
        # Reported only on request.
        ties = choices = 0
        for it in req["items"]:
            valid = np.nonzero((bins - it) >= 0)[0]
            sc = oracle(it, bins[valid])
            k = int(np.argmax(sc))
            choices += int(len(np.unique(bins[valid])) > 1)
            ties += int(len(np.unique(bins[valid][sc >= sc[k]])) > 1)
            b = valid[k]
            order = np.argsort(bins[valid], kind="stable")
            rank = int(np.where(order == k)[0][0]) / max(1, len(valid) - 1)
            bins[b] -= it
            feat.append(round(float(1.0 - bins[b] / capacity), 4))
            feat.append(round(float(rank), 4))
        result = {"ok": True, "value": feat, **({"ties": ties, "choices": choices} if req.get("ties") else {})}
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


# ── constructive TSP (tsp_construct): same two boundaries, another oracle signature ──
# Inner child: exec the candidate ONCE, then serve select_next_node queries.
# Query = (seq, instance, current, destination, n) + n int64 offered node ids;
# answer = (seq, status, chosen). The distance matrices are built once from
# the coordinates in the spec (the candidate sees the full matrix, as in the
# official evaluator).
_CHILD_TSP = r'''
import json, os, struct, sys
import numpy as np
_in = os.fdopen(os.dup(0), "rb")
_out = os.fdopen(os.dup(1), "wb")
_null = os.open(os.devnull, os.O_RDWR)
os.dup2(_null, 0)
os.dup2(_null, 1)
sys.stdin = open(os.devnull, "r")
sys.stdout = open(os.devnull, "w")
spec = json.loads(_in.readline())
if spec.get("eval_seed") is not None:
    import random as _rnd
    _es = int(spec["eval_seed"])
    _rnd.seed(_es)
    np.random.seed(_es % (2 ** 32))
    _orig_default_rng = np.random.default_rng
    _rng_calls = [0]
    def _seeded_default_rng(seed=None, *a, **k):
        if seed is None:
            _rng_calls[0] += 1
            seed = (_es * 1000003 + _rng_calls[0]) % (2 ** 32)
        return _orig_default_rng(seed, *a, **k)
    np.random.default_rng = _seeded_default_rng
coords = [np.asarray(c, dtype=float) for c in spec["coords"]]
mats = [np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2) for c in coords]
ns = {"np": np}
exec(spec["code"], ns)
fn = ns["select_next_node"]
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
    head = _read_exact(20)
    if head is None:
        break
    seq, inst, cur, dest, n = struct.unpack("<iiiii", head)
    raw = _read_exact(8 * n)
    if raw is None:
        break
    try:
        nxt = fn(int(cur), int(dest), np.frombuffer(raw, dtype="<i8").copy(), mats[inst])
        nxt, status = int(nxt), 0
    except Exception:
        nxt, status = 0, -1  # supervisor treats status != 0 as Invalid
    _out.write(struct.pack("<iiq", seq, status, nxt))
    _out.flush()
'''

# Trusted supervisor for tsp_construct: the OFFICIAL construction loop
# (TSPCONST.evaluate_program: nearest-first candidate list of at most
# neighbor_size unvisited nodes, last node forced, closed tour length) runs
# HERE; only select_next_node is delegated to the child. Declared
# differences from the official loop: ties in the distance order are broken
# by city id (stable sort), and the chosen node must be one of the OFFERED
# nodes (the official loop only rejects already-visited nodes).
_HARNESS_TSP = r'''
import json, os, struct, subprocess, sys
import numpy as np

CHILD_SRC = __CHILD_SRC__

def demote():
    os.setgid(65534)
    os.setuid(65534)

class Invalid(Exception):
    pass

class Oracle:
    def __init__(self, code, coords, eval_seed=None):
        self.p = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, preexec_fn=demote)
        self.seq = 0
        self.p.stdin.write((json.dumps({"code": code, "coords": coords, "eval_seed": eval_seed}) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None or struct.unpack("<ii", head) != (0, 0):
            raise Invalid("child failed to initialise candidate")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def __call__(self, inst, cur, dest, offered):
        self.seq += 1
        n = self.seq
        o = np.ascontiguousarray(offered, dtype="<i8").ravel()
        self.p.stdin.write(struct.pack("<iiiii", n, int(inst), int(cur), int(dest), o.shape[0]) + o.tobytes())
        self.p.stdin.flush()
        head = self._read_exact(16)
        if head is None:
            raise Invalid("candidate select call failed")
        seq, status, nxt = struct.unpack("<iiq", head)
        if seq != n or status != 0:
            raise Invalid("candidate select call failed")
        return int(nxt)

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass

req = json.loads(sys.stdin.read())
result = {"ok": False, "error": "unknown op"}
oracle = None
try:
    coords = req["coords"]
    K = int(req.get("neighbor_size", 50))
    want_sig = bool(req.get("signature"))
    oracle = Oracle(req["code"], coords, req.get("eval_seed"))
    lengths, sig, tours = [], [], []
    for k, c in enumerate(coords):
        c = np.asarray(c, dtype=float)
        n = c.shape[0]
        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        nm = np.argsort(D, axis=1, kind="stable")
        route = np.zeros(n, dtype=np.int64)
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        cur = 0
        for i in range(1, n - 1):
            near = nm[cur][1:]
            unv = near[~visited[near]]
            offered = unv[:min(K, unv.size)]
            nxt = oracle(k, cur, 0, offered)
            pos = np.nonzero(offered == nxt)[0]
            if pos.size == 0:
                raise Invalid("selected node not in the offered set")
            if want_sig and k == 0:
                sig.append(round(float((int(pos[0]) + 1) / offered.size), 6))
            route[i] = nxt
            visited[nxt] = True
            cur = nxt
        rem = np.nonzero(~visited)[0]
        if rem.size != 1:
            raise Invalid("tour incomplete")
        route[n - 1] = rem[0]
        lengths.append(float(np.sum(D[route, np.roll(route, -1)])))
        if req.get("return_tours"):
            tours.append([int(x) for x in route])
    result = {"ok": True, "value": float(np.mean(lengths)), "per_instance": lengths,
              **({"signature": sig} if want_sig else {}), **({"tours": tours} if req.get("return_tours") else {})}
except Invalid as exc:
    result = {"ok": False, "error": f"Invalid: {exc}"}
except Exception as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
finally:
    if oracle is not None:
        oracle.close()
print(json.dumps(result))
'''.replace("__CHILD_SRC__", json.dumps(_CHILD_TSP))

# ── TSP crossover design (tsp_ga_crossover): the candidate is a permutation
# crossover; the TRUSTED supervisor runs a fixed GA (initialisation, tournament
# selection, inversion mutation, elitism, evaluation) and asks the child only
# for offspring. Query = (seq, n, seed) + parent1 + parent2 (int64); answer =
# (seq, status) + child (int64, n). Each call gets its own rng seed so an
# evaluation is a deterministic function of (code, GA seed).
_CHILD_TSPGA = r'''
import json, os, struct, sys
import numpy as np
_in = os.fdopen(os.dup(0), "rb")
_out = os.fdopen(os.dup(1), "wb")
_null = os.open(os.devnull, os.O_RDWR)
os.dup2(_null, 0)
os.dup2(_null, 1)
sys.stdin = open(os.devnull, "r")
sys.stdout = open(os.devnull, "w")
spec = json.loads(_in.readline())
import random as _rnd
_es = int(spec.get("eval_seed") or 0)
_rnd.seed(_es)
np.random.seed(_es % (2 ** 32))
coords = [np.asarray(c, dtype=float) for c in spec["coords"]]
mats = [np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2) for c in coords]
ns = {"np": np}
exec(spec["code"], ns)
fn = ns["crossover"]
_out.write(struct.pack("<ii", 0, 0))
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
    head = _read_exact(20)
    if head is None:
        break
    seq, inst, n, seed = struct.unpack("<iiiq", head)
    raw = _read_exact(16 * n)
    if raw is None:
        break
    try:
        par = np.frombuffer(raw, dtype="<i8").copy()
        child = fn(par[:n].copy(), par[n:].copy(), mats[inst], np.random.default_rng(int(seed)))
        arr = np.asarray(child).astype(np.int64).ravel()
        if arr.shape[0] != n:
            raise ValueError("wrong length")
        payload, status = arr.astype("<i8").tobytes(), 0
    except Exception:
        payload, status = b"", -1
    _out.write(struct.pack("<ii", seq, status) + payload)
    _out.flush()
'''

_HARNESS_TSPGA = r'''
import json, os, struct, subprocess, sys, time
import numpy as np

CHILD_SRC = __CHILD_SRC__

def demote():
    os.setgid(65534)
    os.setuid(65534)

class Invalid(Exception):
    pass

class Oracle:
    def __init__(self, code, coords, eval_seed=None):
        self.p = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, preexec_fn=demote)
        self.seq = 0
        self.p.stdin.write((json.dumps({"code": code, "coords": coords, "eval_seed": eval_seed}) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None or struct.unpack("<ii", head) != (0, 0):
            raise Invalid("child failed to initialise candidate")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def __call__(self, inst, p1, p2, seed):
        self.seq += 1
        n = p1.shape[0]
        self.p.stdin.write(struct.pack("<iiiq", self.seq, int(inst), n, int(seed))
                           + np.ascontiguousarray(p1, dtype="<i8").tobytes() + np.ascontiguousarray(p2, dtype="<i8").tobytes())
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None:
            raise Invalid("crossover call failed")
        seq, status = struct.unpack("<ii", head)
        if seq != self.seq or status != 0:
            raise Invalid("crossover call failed")
        raw = self._read_exact(8 * n)
        if raw is None:
            raise Invalid("crossover call failed")
        c = np.frombuffer(raw, dtype="<i8").astype(np.int64)
        if c.shape[0] != n or not np.array_equal(np.sort(c), np.arange(n)):
            raise Invalid("child is not a permutation")
        return c

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass

def edges(t):
    return set(frozenset((int(a), int(b))) for a, b in zip(t, np.roll(t, -1)))

req = json.loads(sys.stdin.read())
result = {"ok": False, "error": "unknown op"}
oracle = None
try:
    coords = req["coords"]
    G = req["ga"]  # {"pop", "gens", "pm", "tournament", "seed"}
    oracle = Oracle(req["code"], coords, req.get("eval_seed"))
    lengths, sig = [], []
    for k, c in enumerate(coords):
        c = np.asarray(c, dtype=float)
        n = c.shape[0]
        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        rng = np.random.default_rng(int(G["seed"]) * 1000 + k)
        P = [rng.permutation(n) for _ in range(int(G["pop"]))]
        F = np.array([float(D[t, np.roll(t, -1)].sum()) for t in P])
        for g in range(int(G["gens"])):
            Q = [P[int(np.argmin(F))]]  # elitism: one
            while len(Q) < int(G["pop"]):
                idx = rng.integers(0, len(P), 4)
                a = P[idx[0]] if F[idx[0]] <= F[idx[1]] else P[idx[1]]
                b = P[idx[2]] if F[idx[2]] <= F[idx[3]] else P[idx[3]]
                child = oracle(k, a, b, int(rng.integers(0, 2**62)))
                if req.get("signature") and k == 0 and g in (0, int(G["gens"]) // 2, int(G["gens"]) - 1) and len(Q) <= 4:
                    ea, eb, ec = edges(a), edges(b), edges(child)
                    sig += [len(ec & ea & eb) / n, len(ec & (ea ^ eb)) / n, len(ec - ea - eb) / n,
                            float(D[child, np.roll(child, -1)].sum() / min(D[a, np.roll(a, -1)].sum(), D[b, np.roll(b, -1)].sum()))]
                if rng.random() < float(G["pm"]):
                    x, y = sorted(rng.integers(0, n, 2))
                    child = child.copy()
                    child[x:y + 1] = child[x:y + 1][::-1]
                Q.append(child)
            P = Q
            F = np.array([float(D[t, np.roll(t, -1)].sum()) for t in P])
        lengths.append(float(F.min()))
    result = {"ok": True, "value": float(np.mean(lengths)), "per_instance": lengths,
              **({"signature": sig} if req.get("signature") else {})}
except Invalid as exc:
    result = {"ok": False, "error": f"Invalid: {exc}"}
except Exception as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
finally:
    if oracle is not None:
        oracle.close()
print(json.dumps(result))
'''.replace("__CHILD_SRC__", json.dumps(_CHILD_TSPGA))

# ── tspga2: operator suite (L2) or crossover only (L1) with a COMMON PARENT-PAIR
# PROBE measured in a separate isolated child process before the GA. Query head
# = (seq, instance, op, n, seed); op 1 = crossover (payload 2n ids), op 2 =
# mutate (n ids), op 3 = local_improve (n ids); answer = (seq, status) + n ids.
_CHILD_TSPGA2 = r'''
import json, os, struct, sys
import numpy as np
_in = os.fdopen(os.dup(0), "rb")
_out = os.fdopen(os.dup(1), "wb")
_null = os.open(os.devnull, os.O_RDWR)
os.dup2(_null, 0)
os.dup2(_null, 1)
sys.stdin = open(os.devnull, "r")
sys.stdout = open(os.devnull, "w")
spec = json.loads(_in.readline())
import random as _rnd
_es = int(spec.get("eval_seed") or 0)
_rnd.seed(_es)
np.random.seed(_es % (2 ** 32))
coords = [np.asarray(c, dtype=float) for c in spec["coords"]]
mats = [np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2) for c in coords]
ns = {"np": np}
exec(spec["code"], ns)
fns = {1: ns.get("crossover"), 2: ns.get("mutate"), 3: ns.get("local_improve")}
_out.write(struct.pack("<ii", 0, 0))
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
    head = _read_exact(24)
    if head is None:
        break
    seq, inst, op, n, seed = struct.unpack("<iiiiq", head)
    raw = _read_exact(8 * (2 * n if op == 1 else n))
    if raw is None:
        break
    try:
        arr_in = np.frombuffer(raw, dtype="<i8").copy()
        fn = fns[op]
        if fn is None:
            raise ValueError("missing function")
        g = np.random.default_rng(int(seed))
        if op == 1:
            out = fn(arr_in[:n].copy(), arr_in[n:].copy(), mats[inst], g)
        else:
            out = fn(arr_in.copy(), mats[inst], g)
        arr = np.asarray(out).astype(np.int64).ravel()
        if arr.shape[0] != n:
            raise ValueError("wrong length")
        payload, status = arr.astype("<i8").tobytes(), 0
    except Exception:
        payload, status = b"", -1
    _out.write(struct.pack("<ii", seq, status) + payload)
    _out.flush()
'''

_HARNESS_TSPGA2 = r'''
import json, os, struct, subprocess, sys
import numpy as np

CHILD_SRC = __CHILD_SRC__

def demote():
    os.setgid(65534)
    os.setuid(65534)

class Invalid(Exception):
    pass

class Oracle:
    def __init__(self, code, coords, eval_seed=None):
        self.p = subprocess.Popen(
            [sys.executable, "-I", "-c", CHILD_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, preexec_fn=demote)
        self.seq = 0
        self.p.stdin.write((json.dumps({"code": code, "coords": coords, "eval_seed": eval_seed}) + "\n").encode("utf-8"))
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None or struct.unpack("<ii", head) != (0, 0):
            raise Invalid("child failed to initialise candidate")

    def _read_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def __call__(self, inst, op, a, b, seed):
        self.seq += 1
        n = a.shape[0]
        payload = np.ascontiguousarray(a, dtype="<i8").tobytes()
        if op == 1:
            payload += np.ascontiguousarray(b, dtype="<i8").tobytes()
        self.p.stdin.write(struct.pack("<iiiiq", self.seq, int(inst), int(op), n, int(seed)) + payload)
        self.p.stdin.flush()
        head = self._read_exact(8)
        if head is None:
            raise Invalid("operator call failed")
        seq, status = struct.unpack("<ii", head)
        if seq != self.seq or status != 0:
            raise Invalid("operator call failed")
        raw = self._read_exact(8 * n)
        if raw is None:
            raise Invalid("operator call failed")
        c = np.frombuffer(raw, dtype="<i8").astype(np.int64)
        if c.shape[0] != n or not np.array_equal(np.sort(c), np.arange(n)):
            raise Invalid("result is not a permutation")
        return c

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass

def tour_len(t, D):
    return float(D[t, np.roll(t, -1)].sum())

def edges(t):
    return set(frozenset((int(a), int(b))) for a, b in zip(t, np.roll(t, -1)))

def two_opt(tour, D, cap_factor=20, tol=1e-9):
    t = list(int(x) for x in tour); n = len(t); cap = n * n * cap_factor; evals = 0; improved = True
    while improved and evals < cap:
        improved = False
        for i in range(0, n - 2):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                evals += 1
                if D[a, c] + D[b, d] - D[a, b] - D[c, d] < -tol:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]; improved = True
                if evals >= cap:
                    break
            if evals >= cap:
                break
    return np.array(t, dtype=np.int64)

def nn_tour(D, start):
    n = D.shape[0]; visited = np.zeros(n, dtype=bool); visited[start] = True; t = [start]; cur = start
    for _ in range(n - 1):
        d = D[cur].copy(); d[visited] = np.inf; cur = int(np.argmin(d)); visited[cur] = True; t.append(cur)
    return np.array(t, dtype=np.int64)

def probe_pairs(D):
    """Fixed common parent pairs on instance 0: 4 random-random, 4 NN-NN (different starts),
    4 improved-improved (budgeted 2-opt of random tours), 4 mixed (both orders)."""
    n = D.shape[0]; g = np.random.default_rng(777)
    rnd = [g.permutation(n) for _ in range(12)]
    nn = [nn_tour(D, s) for s in range(8)]
    opt = [two_opt(r, D) for r in rnd[8:12]] + [two_opt(nn[s], D) for s in (0, 1, 2, 3)]
    pairs = [(rnd[0], rnd[1]), (rnd[2], rnd[3]), (rnd[4], rnd[5]), (rnd[6], rnd[7]),
             (nn[0], nn[1]), (nn[2], nn[3]), (nn[4], nn[5]), (nn[6], nn[7]),
             (opt[0], opt[1]), (opt[2], opt[3]), (opt[4], opt[5]), (opt[6], opt[7]),
             (nn[0], opt[0]), (opt[1], nn[1]), (rnd[0], nn[2]), (opt[2], rnd[1])]
    return pairs

def pipeline(oracle, k, a, b, seeds, pipe, pm, rng_draw):
    c = oracle(k, 1, a, b, seeds[0])
    if pipe == "l2":
        if rng_draw < pm:
            c = oracle(k, 2, c, None, seeds[1])
        c = oracle(k, 3, c, None, seeds[2])
    else:
        if rng_draw < pm:
            n = c.shape[0]; g = np.random.default_rng(int(seeds[1])); x, y = sorted(g.integers(0, n, 2))
            c = c.copy(); c[x:y + 1] = c[x:y + 1][::-1]
    return c

req = json.loads(sys.stdin.read())
result = {"ok": False, "error": "unknown op"}
oracle = None
try:
    coords = req["coords"]
    G = req["ga"]
    pipe = req.get("pipeline", "l1")
    pm = float(G["pm"])
    lengths, sig, tours = [], [], []
    if req.get("signature"):
        c0 = np.asarray(coords[0], dtype=float)
        D0 = np.linalg.norm(c0[:, None, :] - c0[None, :, :], axis=2)
        n0 = D0.shape[0]
        probe = Oracle(req["code"], coords, req.get("eval_seed"))  # ISOLATED process for the probe
        try:
            for i, (a, b) in enumerate(probe_pairs(D0)):
                # probe draw: l1 = crossover only (no fixed inversion: draw 1.0 >= pm); l2 = forced three-stage (declared)
                child = pipeline(probe, 0, a, b, (9000 + 3 * i, 9001 + 3 * i, 9002 + 3 * i), pipe, pm, 1.0 if pipe == "l1" else -1.0)
                ea, eb, ec = edges(a), edges(b), edges(child)
                sig += [len(ec & ea & eb) / n0, len(ec & (ea ^ eb)) / n0, len(ec - ea - eb) / n0,
                        tour_len(child, D0) / min(tour_len(a, D0), tour_len(b, D0))]
        finally:
            probe.close()
    oracle = Oracle(req["code"], coords, req.get("eval_seed"))  # fresh process for the GA
    for k, c in enumerate(coords):
        c = np.asarray(c, dtype=float)
        n = c.shape[0]
        D = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        rng = np.random.default_rng(int(G["seed"]) * 1000 + k)
        P = [rng.permutation(n) for _ in range(int(G["pop"]))]
        F = np.array([tour_len(t, D) for t in P])
        for g in range(int(G["gens"])):
            Q = [P[int(np.argmin(F))]]
            while len(Q) < int(G["pop"]):
                idx = rng.integers(0, len(P), 4)
                a = P[idx[0]] if F[idx[0]] <= F[idx[1]] else P[idx[1]]
                b = P[idx[2]] if F[idx[2]] <= F[idx[3]] else P[idx[3]]
                seeds = rng.integers(0, 2**62, 3)
                Q.append(pipeline(oracle, k, a, b, seeds, pipe, pm, float(rng.random())))
            P = Q
            F = np.array([tour_len(t, D) for t in P])
        lengths.append(float(F.min()))
        if req.get("return_tours"):
            tours.append([int(x) for x in P[int(np.argmin(F))]])
    result = {"ok": True, "value": float(np.mean(lengths)), "per_instance": lengths,
              **({"signature": sig} if req.get("signature") else {}), **({"tours": tours} if req.get("return_tours") else {})}
except Invalid as exc:
    result = {"ok": False, "error": f"Invalid: {exc}"}
except Exception as exc:
    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
finally:
    if oracle is not None:
        oracle.close()
print(json.dumps(result))
'''.replace("__CHILD_SRC__", json.dumps(_CHILD_TSPGA2))

HARNESSES = {"bp": _HARNESS, "tsp": _HARNESS_TSP, "tspga": _HARNESS_TSPGA, "tspga2": _HARNESS_TSPGA2}


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=20, creationflags=_NOWIN).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def ensure_image() -> None:
    """Build the pinned sandbox image if absent (network used at BUILD only)."""
    have = subprocess.run(["docker", "image", "inspect", IMAGE],
                          capture_output=True, creationflags=_NOWIN)
    if have.returncode == 0:
        return
    dockerfile = "FROM python:3.11-slim\nRUN pip install --no-cache-dir numpy==2.4.4\n"
    r = subprocess.run(["docker", "build", "-t", IMAGE, "-"],
                       input=dockerfile.encode("utf-8"), capture_output=True,
                       timeout=600, creationflags=_NOWIN)
    if r.returncode != 0:
        raise RuntimeError(f"sandbox image build failed: {r.stderr.decode()[-400:]}")


def image_digest() -> "str | None":
    """Docker image ID of the sandbox (recorded in run reports;). With the remote backend the
    record names the server URL instead, so a cohort's backend is visible in every design.json."""
    import os  # noqa: PLC0415
    if os.environ.get("TGADE_EVAL_REMOTE_URL"):
        return "remote:" + os.environ["TGADE_EVAL_REMOTE_URL"]
    if os.environ.get("TGADE_EVAL_BACKEND") == "linux":
        import platform  # noqa: PLC0415
        return "linux-direct:" + platform.node()
    global _image_digest_cache
    if _image_digest_cache is None:
        r = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                          capture_output=True, text=True, creationflags=_NOWIN)
        if r.returncode == 0:
            _image_digest_cache = r.stdout.strip()
    return _image_digest_cache


def _run_remote(payload: dict, timeout: int, harness: str) -> "dict | None":
    """Fallback backend (insurance against a local failure): POST the trusted
    supervisor source and the payload to a key-less evaluation server (local/colab_eval/server.py,
    e.g. on a Colab VM) that runs `python -I -c <harness>` with the same demoted child. The
    server holds no secrets; a cohort must run entirely on ONE backend (recorded by image_digest)."""
    import os  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    url = os.environ["TGADE_EVAL_REMOTE_URL"].rstrip("/") + "/eval"
    body = json.dumps({"harness_src": HARNESSES[harness], "payload": payload, "timeout": timeout}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "X-Eval-Token": os.environ.get("TGADE_EVAL_REMOTE_TOKEN", "")})
    try:
        with urllib.request.urlopen(req, timeout=timeout + 60) as r:
            out = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001  (transport failure = missing measurement, never a value)
        return None
    if out.get("timeout") or out.get("returncode") != 0:
        return None
    lines = (out.get("stdout") or "").strip().splitlines()
    if len(lines) != 1:
        return None
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError:
        return None


def _run_linux(payload: dict, timeout: int, harness: str) -> "dict | None":
    """Backend for a disposable Linux VM without Docker (Colab, 2026-09-28): the trusted supervisor
    runs directly as the current (root) user with a CLEAN environment, so the runner's API key
    (kept only in the runner's own environment) is never visible to it; the supervisor demotes the
    candidate to uid 65534 as inside Docker. No network isolation: use only on a VM that holds
    nothing but this job. Refused on non-POSIX hosts."""
    import os  # noqa: PLC0415
    if os.name != "posix":
        return None
    try:
        r = subprocess.run([__import__("sys").executable, "-I", "-c", HARNESSES[harness]], input=json.dumps(payload),
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"})
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    lines = r.stdout.strip().splitlines()
    if len(lines) != 1:
        return None
    try:
        return json.loads(lines[0])
    except json.JSONDecodeError:
        return None


def run_sandboxed(payload: dict, timeout: int, harness: str = "bp") -> "dict | None":
    """One candidate execution. Returns the supervisor JSON or None
    (fail-closed on timeout, abnormal exit, or malformed record).
    harness: "bp" (online bin packing, score oracle) or "tsp" (constructive TSP)."""
    import os  # noqa: PLC0415
    if os.environ.get("TGADE_EVAL_REMOTE_URL"):
        return _run_remote(payload, timeout, harness)
    if os.environ.get("TGADE_EVAL_BACKEND") == "linux":
        return _run_linux(payload, timeout, harness)
    _HARNESS = HARNESSES[harness]
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
                           timeout=timeout, creationflags=_NOWIN)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, creationflags=_NOWIN)
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
