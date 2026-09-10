"""EoH online-bin-packing adapter for the bench engine (G2c E2).

Substrate = the OFFICIAL EoH repository (third_party/EoH, MIT): its task
description, prompt templates (i1/e1/e2/m1/m2), code extractor and evaluator
logic are used as OFFICIAL TEMPLATES WITH DECLARED WRAPPERS: a
"/no_think" prefix on every prompt, one strong-mutation instruction line, and
a custom integrity-repair prompt. Identical wrappers and budgets must be used
across matched arms; published EoH numbers are descriptive only unless their
protocol is matched. The vendored Weibull 5x5000 files are named a "test"
dataset upstream but serve HERE as the evolution/training bank (dev subsets
are development data); the E4 confirmation bank is generated separately. The
evolution loop, ledger, and certification live in the engine.

Genotype  genes = {"thought": <NL algorithm description>, "code": <score fn>}.
          thought is the inherited natural-language description
          (thought-conditioning).
Energy    official evaluator on the TRAIN set (5 Weibull instances x 5000,
          capacity 100); raw = mean excess over the L1 bound, >= 0. E =
          min(raw, RAW_MAX) / RAW_MAX with RAW_MAX = 2.0 (declared linear
          clip, C5; worst-fit ~ 1.51 stays inside the ramp).
Diversity behaviour signature: per-step (fill ratio, tightness rank) of the
          chosen bin on a fixed TRAIN-derived probe (first PROBE_LEN items of
          training instance 0), flattened to one (1, 2*PROBE_LEN) row.
          Offline and deterministic — no embedding cost.
Step 6    integrity(genes): the individual's host client repairs syntax/
          format of the code ONLY (thought untouched); extraction failure =>
          None (fail-closed vacancy).
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from grant_evo.tgade.engine import LLMClient

_REPO = Path(__file__).resolve().parents[4]
_EOH_SRC = _REPO / "third_party" / "EoH" / "eoh" / "src"
_BP = _REPO / "third_party" / "EoH" / "examples" / "bp_online"
for p in (str(_EOH_SRC), str(_BP)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _lazy_imports():
    # eoh.llm.api_local_llm imports `requests` at package-import time; that
    # transport is never used here (the engine injects clients). Stub it when
    # absent so the OFFICIAL prompts/extractor/evaluator import offline.
    try:  # noqa: SIM105
        import requests  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        import types  # noqa: PLC0415
        stub = types.ModuleType("requests")

        def _unavailable(*a, **k):  # pragma: no cover - never called offline
            raise RuntimeError("requests stub: local-LLM transport unused")

        stub.post = _unavailable
        stub.get = _unavailable
        sys.modules["requests"] = stub
    from eoh.config import EoHConfig, LLMConfig  # noqa: PLC0415
    from eoh.eoh import evolution as EV  # noqa: PLC0415
    from prob import BPONLINE  # noqa: PLC0415

    class _StubLLM:  # Evolution() pings its client at construction; stub it.
        def __init__(self, *a, **k):
            pass

        def get_response(self, prompt):
            return "stub"

    EV.InterfaceLLM = _StubLLM
    return EV, EoHConfig, LLMConfig, BPONLINE


RAW_MAX = 2.0

# Determinism gate: scoring functions that draw random numbers get a lucky-draw
# advantage under best-value-keeping selection and take over lineages (up to 65 % of candidates in
# some runs). With deterministic_only=True such candidates are invalid (energy None) BEFORE any
# sandbox run. Static check on the comment/docstring-free, ast-unparsed source.
RANDOM_PATTERN = re.compile(r"\brandom\b|\brandint\b|\brand\(|\bnormal\(|\buniform\(|\bshuffle\(|\bchoice\(|"
                            r"\bsecrets\b|\burandom\b|\bdefault_rng\b|\bRandomState\b|\bpermutation\(")


def uses_randomness(code: str) -> bool:
    try:
        src = BpOnlineAdapter.normalise_code(code)
    except Exception:  # noqa: BLE001  (unparsable code is judged on its raw text)
        src = code
    return bool(RANDOM_PATTERN.search(src))
PROBE_LEN = 64
NL_EMBED_MODEL = "text-embedding-3-small"
NL_EMBED_DIM = 1536
EVAL_TIMEOUT = 60


# Step-6 description-only repair prompt (single source for the live path and
# scripts/r_static_diagnosis.py). Revision a2 (G-2, 2026-09-06): the first
# version omitted the scoring semantics, so Qwen described the arithmetic
# without a preference direction (7/10 on the minimal set); a2 states the
# contract and demands the direction explicitly.
ALIGN_THOUGHT_PROMPT = (
    "/no_think\nBelow is a Python scoring function for online bin packing: `bins` holds the remaining capacities of the "
    "feasible bins, `item` is the item size, and the bin with the HIGHEST score receives the item. Write ONE sentence, "
    "inside braces {}, that states (1) the scoring principle in words, (2) which bins score higher: smaller remaining "
    "capacity (tight fit), larger remaining capacity (loose fit), or another criterion, and (3) any bonus, threshold, "
    "index/order effect or tie rule. Describe the code as it is, even if it looks wrong; do not add intent or praise. "
    "Output only the braced sentence.\n\n")


class BpOnlineAdapter:
    name = "bp_online"
    num_sections = 1
    dim = 2 * PROBE_LEN
    # oracle replay predicate (conservative superset of the EoH extractor:
    # a {...} description span + a ```python block containing def score)
    parse_predicate = "bp_online_v1"
    energy_spec = f"E = min(raw_excess, {RAW_MAX}) / {RAW_MAX} (declared linear clip)"
    diversity_spec = (f"behaviour signature: (fill, tightness-rank) per step on the "
                      f"first {PROBE_LEN} items of TRAIN instance 0, L2 unit rows (raw rows before 2026-09-07)")

    def __init__(self, train_k: int = 5, items: int = 5000, seed: int = 0,
                 mutation_template: str = "m1", diversity_carrier: str = "behaviour",
                 objective: str = "train_c100", integrity_kind: str = "syntax",
                 hybrid_weights: "tuple | None" = None, probe_len: int = PROBE_LEN,
                 eval_seed: "int | None" = None, eval_timeout: "float | None" = None,
                 signature_source: str = "probe", deterministic_only: bool = False):
        EV, EoHConfig, LLMConfig, BPONLINE = _lazy_imports()
        self._ev = EV
        # V101 D3: which official EoH mutation template Step 5 uses. "m1" =
        # new algorithm of a different form (canonical so far); "m2" = keep the
        # algorithm, change its parameter settings (local adjustment). The
        # sampling-temperature ladder applies to either. Declared operator
        # difference: recorded via operator_spec in the run manifest.
        # "mix": EoH operator composition inside Algorithm 1 -
        # each Step-5 mutation call draws m1 (form change) or m2 (parameter
        # tuning) with equal probability from the engine rng.
        if mutation_template not in ("m1", "m2", "mix"):
            raise ValueError(f"unknown mutation_template: {mutation_template!r}")
        self.mutation_template = mutation_template
        # Diversity: which carrier feeds log det. "behaviour"
        # = sandbox probe signature (canonical so far); "nl" = fixed-model
        # embedding of the thought/description text (genotype), unit rows.
        # Stage-A gate result (nlgate_gateA_contrast26*): paraphrase share of
        # the distinct-rule reward ~1.0, i.e. the NL carrier separates
        # duplicates from non-duplicates but NOT paraphrase from mechanism.
        # "hybrid": concat of L2-unit blocks
        # [sqrt(w_bh) behaviour, sqrt(w_desc) description, sqrt(w_code) normalised-code]
        # with hybrid_weights (bh, desc, code) summing to 1; a zero weight drops the block.
        if diversity_carrier not in ("behaviour", "nl", "hybrid"):
            raise ValueError(f"unknown diversity_carrier: {diversity_carrier!r}")
        self.diversity_carrier = diversity_carrier
        hw = tuple(float(x) for x in (hybrid_weights or (0.8, 0.1, 0.1)))
        if len(hw) != 3 or any(x < 0 for x in hw) or abs(sum(hw) - 1.0) > 1e-9:
            raise ValueError(f"hybrid_weights must be 3 non-negative numbers summing to 1, got {hw!r}")
        self.hybrid_weights = hw

        # objective "train_c100" = canonical E4 train set (C=100, 5 instances);
        # "regime_max" = robust two-capacity objective Q = max(mean excess on the
        # sealed regime train bank C=100, mean excess on the C=500 bank), raw
        # then clipped like energy(). Components are kept in self.last_components.
        if objective not in ("train_c100", "regime_max"):
            raise ValueError(f"unknown objective: {objective!r}")
        self.objective = objective
        # Step 6 integrity: the repair the host model must own is
        # description<->code coherence, not syntax. "syntax" = legacy syntax-only
        # code repair (thought untouched); "align_thought" = the individual's host
        # rewrites the description to match the code; "align_code" = the host makes
        # the code implement the description.
        # "align_thought_meas" (replan round 2026-09-06, R-meas): the repaired
        # description is stored in genes["thought_meas"] and feeds ONLY the NL
        # diversity carrier; genes["thought"] (what operators see) and the
        # loci digest stay untouched. Repairs are cached by code sha.
        if integrity_kind not in ("syntax", "align_thought", "align_thought_meas", "align_code"):
            raise ValueError(f"unknown integrity_kind: {integrity_kind!r}")
        self.integrity_kind = integrity_kind
        self.integrity_fallbacks = 0
        # Evaluation-seed panel: None = historical evaluator (candidate RNG
        # uninitialised); an int seeds random/np.random inside every sandbox request (energy: seed,
        # signature: seed+1, endpoint/regime: seed+2) so scores are functions of (code, seed).
        self.eval_seed = None if eval_seed is None else int(eval_seed)
        # Lethal-candidate threshold: wall-clock cap of one train evaluation
        # (5 instances) and of the signature probe; a candidate over the cap is None (never a value).
        # Default keeps the historical 120 s; Stage 1 v2 T>0 arms use 30 s (recorded in design.json).
        self.eval_timeout = float(EVAL_TIMEOUT + 60) if eval_timeout is None else float(eval_timeout)
        self._repair_cache: dict = {}
        # single-flight per cache key (repair / embedding) so concurrent callers with the
        # same code or text make ONE call and share the result.
        import threading as _th  # noqa: PLC0415
        self._keylocks: dict = {}
        self._keylocks_guard = _th.Lock()
        self.last_components: dict = {}
        self._regime = None
        if objective == "regime_max":
            self._regime = self._load_regime_banks()
            self.energy_spec = ("E = min(max(raw_excess@C100_train_bank, raw_excess@C500_train_bank), "
                                f"{RAW_MAX}) / {RAW_MAX}; banks experiments/e4 (sealed)")
        self._embedder = None
        self._embed_cache: dict = {}
        if diversity_carrier == "hybrid":
            wb, wd, wc = self.hybrid_weights
            self.num_sections, self.dim = 1, (2 * PROBE_LEN if wb > 0 else 0) + (NL_EMBED_DIM if wd > 0 else 0) + (NL_EMBED_DIM if wc > 0 else 0)
            self.diversity_spec = (f"hybrid neighborhood: concat of L2-unit blocks [sqrt({wb}) behaviour signature ({2 * PROBE_LEN}-d), "
                                   f"sqrt({wd}) description embedding {NL_EMBED_MODEL} ({NL_EMBED_DIM}-d), sqrt({wc}) normalised-code embedding "
                                   f"({NL_EMBED_DIM}-d, comments/docstrings stripped, ast-unparsed)], unit row; zero-weight blocks dropped")
        if diversity_carrier == "nl":
            self.num_sections, self.dim = 1, NL_EMBED_DIM
            self.diversity_spec = (f"nl embedding: {NL_EMBED_MODEL} ({NL_EMBED_DIM}-d, L2 unit row) of the "
                                   f"thought/description text (whitespace-normalised, sha-cached); fixed model; "
                                   f"behaviour signature not used for selection")
        self.operator_spec = (f"init=i1; cross=e1|e2 (random); mutate={mutation_template} x sampling "
                              f"temperature {self.STRENGTH_TEMPERATURE}; integrity={integrity_kind}; "
                              f"prompt prefix /no_think")
        self.problem = BPONLINE(capacity=100, timeout=EVAL_TIMEOUT)
        # No genotype memo: a memo would make a candidate that
        # uses unseeded randomness look deterministic across re-evaluations
        # (cache hit vs fresh measurement differ). Every viability check is a
        # fresh sandbox run; elite re-evaluation costs one run per generation.
        if train_k < 5 or items < 5000:  # dev subset (development data only)
            r = random.Random(seed)
            for dsname, ds in self.problem.instances.items():
                keys = sorted(ds)
                keep = sorted(r.sample(keys, min(train_k, len(keys))))
                sub = {}
                for k in keep:
                    inst = dict(ds[k])
                    inst["items"] = list(inst["items"])[:items]
                    inst["num_items"] = len(inst["items"])
                    sub[k] = inst
                self.problem.instances[dsname] = sub
                from get_instance import GetData  # noqa: PLC0415
                self.problem.lb[dsname] = GetData().l1_bound_dataset(sub)
        cfg = EoHConfig(llm=LLMConfig(api_endpoint="stub", api_key="stub", model="stub"),
                        pop_size=4, n_pop=1, operators=["e1", "e2", "m1", "m2"])
        self.evo = EV.Evolution(cfg, self.problem)  # prompts + extractor ONLY
        ds = next(iter(self.problem.instances.values()))
        # V3: a longer probe separates mechanisms the 64-item
        # probe conflates (e0359/p0366 differ from item 341). probe_len is recorded in the spec.
        self.probe_len = int(probe_len)
        self._probe = list(next(iter(ds.values()))["items"])[:self.probe_len]
        # Signature source: "probe" = separate sandbox run on the first probe_len
        # items of train instance 0 (historical); "energy" = by-product of the energy evaluation itself,
        # (fill, tightness-rank) of EVERY step of train instance 0, no extra execution.
        if signature_source not in ("probe", "energy"):
            raise ValueError(f"unknown signature_source: {signature_source!r}")
        if signature_source == "energy" and objective != "train_c100":
            raise ValueError("signature_source='energy' needs the train_c100 objective")
        self.signature_source = signature_source
        self._sig_cache: "dict[str, np.ndarray | None]" = {}
        self.deterministic_only = bool(deterministic_only)
        self.randomness_rejects = 0
        if signature_source == "energy":
            self._probe = list(next(iter(ds.values()))["items"])
            self.probe_len = len(self._probe)
            if self.diversity_carrier in ("behaviour", "hybrid"):
                self.diversity_spec += "; signature = by-product of the energy evaluation (all steps of train instance 0)"
        # declared row length follows the ACTUAL probe (dev subsets can be shorter than PROBE_LEN)
        _bh_dim = 2 * len(self._probe)
        if self.diversity_carrier in ("behaviour", "hybrid"):
            self.diversity_spec = self.diversity_spec.replace(f"first {PROBE_LEN} items", f"first {len(self._probe)} items")
        if self.diversity_carrier == "behaviour":
            self.dim = _bh_dim
        elif self.diversity_carrier == "hybrid":
            wb, wd, wc = self.hybrid_weights
            self.dim = (_bh_dim if wb > 0 else 0) + (NL_EMBED_DIM if wd > 0 else 0) + (NL_EMBED_DIM if wc > 0 else 0)

    # ── genotype plumbing ────────────────────────────────────────────
    def _extract(self, text: "str | None"):
        if not text:
            return None
        algorithm, code = self.evo._extract(text)
        if not algorithm or not code:
            return None
        return {"thought": " ".join(algorithm[0].split())[:2000],
                "code": self.evo._prepend_imports(code[0])}

    def loci_view(self, genes: dict) -> dict:
        return {"thought": genes["thought"], "code": genes["code"]}

    def validate(self, genes: dict) -> tuple[bool, str]:
        if not genes.get("thought", "").strip():
            return False, "empty thought"
        code = genes.get("code", "")
        if "def score" not in code:
            return False, "no score function"
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"syntax: {exc}"
        return True, "ok"

    # ── pure parser for the E3 gate ─────────────
    def reconstruct(self, role: str, text, parents: list, strength: str = "mid"):
        """Reconstruct the genotype an operator would have produced from the
        retained raw response: every bp_online operator (init/mutate/cross)
        is `_extract(text)`; integrity keeps the parent thought and
        re-extracts the code block. Pure: no LLM, no randomness."""
        if text is None:
            return None
        if role in ("init", "mutate", "cross"):
            return self._extract(text)
        if role == "integrity":
            if self.integrity_kind in ("align_thought", "align_thought_meas"):
                return self._align_thought_from_text(text, parents[0])
            _, code = self.evo._extract(text)
            if not code:
                return None
            return {"thought": parents[0]["thought"],
                    "code": self.evo._prepend_imports(code[0])}
        raise ValueError(role)


    # align_thought path and its replay reconstruction. Description-only
    # repair: the code bytes are the input's; a response without a braced
    # sentence falls back to the ORIGINAL genes (never a vacancy) and is
    # counted in integrity_fallbacks.
    def _keylock(self, kind: str, key: str):
        import threading as _th  # noqa: PLC0415
        with self._keylocks_guard:
            return self._keylocks.setdefault((kind, key), _th.Lock())

    def _align_thought_cached(self, ck: str, genes: dict, llm) -> dict:
        """Single-flight body of the description-only repair (caller holds the
        per-code key lock): cache hit -> no call; else ONE call, cached either way."""
        import re as _re  # noqa: PLC0415
        if ck in self._repair_cache:
            if self._repair_cache[ck] is None:
                self.integrity_fallbacks += 1
            return self._align_result(genes, self._repair_cache[ck])
        prompt = ALIGN_THOUGHT_PROMPT + genes["code"]
        # repair is a deterministic re-description, so it
        # runs at sampling temperature 0 (the static G-2 diagnosis setting), not
        # the operator client default (0.8). Runs before this fix used 0.8.
        # Content-derived logical id: the provider seed of a repair depends on the code
        # and the repair contract only, never on which operator reached the cache first.
        out = llm(prompt, temperature=0.0, call_id=f"repair-a2-{self.integrity_kind}-{ck[:32]}")
        text = out if isinstance(out, str) else (getattr(out, "text", None) or "")
        m = _re.search(r"\{(.*?)\}", text or "", _re.S)
        # cache the outcome either way: a recurring code that
        # yielded no braced sentence falls back without another call.
        self._repair_cache[ck] = " ".join(m.group(1).split()) if (m and m.group(1).strip()) else None
        return self._align_thought_from_text(text, genes)

    def _align_thought_from_text(self, text, genes: dict) -> dict:
        import re as _re  # noqa: PLC0415
        m = _re.search(r"\{(.*?)\}", text or "", _re.S)
        if not m or not m.group(1).strip():
            self.integrity_fallbacks += 1
            return self._align_result(genes, None)
        return self._align_result(genes, " ".join(m.group(1).split()))

    def _align_result(self, genes: dict, repaired) -> dict:
        """R-gen (align_thought): repaired text replaces thought. R-meas
        (align_thought_meas): thought unchanged, repaired text in thought_meas.
        None = fallback to the original description; code bytes always kept."""
        if self.integrity_kind == "align_thought_meas":
            return {"thought": genes["thought"], "code": genes["code"],
                    "thought_meas": repaired if repaired is not None else genes.get("thought_meas", genes["thought"])}
        return {"thought": repaired if repaired is not None else genes["thought"], "code": genes["code"]}

    # ── operators (EoH prompts verbatim; engine injects the clients) ─
    def init_genes(self, rng: random.Random, llm: LLMClient) -> "dict | None":
        return self._extract(llm("/no_think\n" + self.evo._build_prompt("i1")))

    # M2 strength ladder v2: a single-dimension
    # SAMPLING-TEMPERATURE dial on the official m1 template. v1 (weak=m2,
    # mid=m1, strong=m1 + "structurally different" line) measured NON-
    # monotone behavioural diversity on qwen3-32b (entropy weak .59 > mid
    # .35 > strong .28, E3 run 20260904T113215): the extra instruction
    # collapsed outputs onto one canonical alternative. Temperature is a
    # genuine perturbation-strength parameter; it is sent per call and
    # recorded in the ledger (opts). Declared wrapper; identical for all arms.
    STRENGTH_TEMPERATURE = {"weak": 0.5, "mid": 0.8, "strong": 1.1}

    def mutate(self, genes: dict, strength: str, rng: random.Random,
               llm: LLMClient, template: "str | None" = None) -> "dict | None":
        parent = {"algorithm": genes["thought"], "code": genes["code"]}
        if template is None:  # explicit template = EoH operator policy (EoH-style offspring policy); no rng draw then
            template = rng.choice(["m1", "m2"]) if self.mutation_template == "mix" else self.mutation_template
        prompt = self.evo._build_prompt(template, parent)
        temperature = self.STRENGTH_TEMPERATURE[strength]
        return self._extract(llm("/no_think\n" + prompt, temperature=temperature))

    def crossover(self, a: dict, b: dict, rng: random.Random,
                  llm: LLMClient, template: "str | None" = None) -> "dict | None":
        parents = [{"algorithm": a["thought"], "code": a["code"]},
                   {"algorithm": b["thought"], "code": b["code"]}]
        prompt = self.evo._build_prompt(template or rng.choice(["e1", "e2"]), parents)
        return self._extract(llm("/no_think\n" + prompt))

    def integrity(self, genes: dict, rng: random.Random,
                  llm: LLMClient) -> "dict | None":
        import re as _re  # noqa: PLC0415
        if self.integrity_kind in ("align_thought", "align_thought_meas"):
            ck = hashlib.sha256(genes["code"].encode("utf-8")).hexdigest()
            with self._keylock("repair", ck):
                return self._align_thought_cached(ck, genes, llm)
        if self.integrity_kind == "align_code":
            prompt = ("/no_think\nThe description below states the intended scoring rule; the code must implement exactly that rule. "
                      "If the code already implements it, return it unchanged; otherwise minimally change the code so that it does. "
                      "Keep the function name, inputs and outputs; return the full function in a ```python code block.\n\n"
                      f"Description: {genes['thought']}\n\nCode:\n{genes['code']}")
            out = llm(prompt)
            text = out if isinstance(out, str) else (getattr(out, "text", None) or "")
            _, code = self.evo._extract(text)
            if not code:
                return None
            return {"thought": genes["thought"], "code": self.evo._prepend_imports(code[0])}
        prompt = ("/no_think\nRepair ONLY syntax/format problems in this Python "
                  "function; keep the algorithm identical; keep the name, inputs "
                  "and outputs unchanged; return the full corrected function in "
                  "a ```python code block.\n\n" + genes["code"])
        out = llm(prompt)
        if out is None:
            return None
        _, code = self.evo._extract(out)
        if not code:
            return None
        return {"thought": genes["thought"],
                "code": self.evo._prepend_imports(code[0])}

    # ── measurement (SANDBOXED: candidate code never runs on host;
    #) ────────────────────────────────────────────────
    def _instances_payload(self) -> tuple[dict, dict]:
        inst = {ds: {str(k): {"capacity": v["capacity"],
                              "num_items": v["num_items"],
                              "items": [int(x) for x in v["items"]]}
                     for k, v in d.items()}
                for ds, d in self.problem.instances.items()}
        return inst, {k: float(v) for k, v in self.problem.lb.items()}

    @staticmethod
    def _load_regime_banks() -> dict:
        import hashlib  # noqa: PLC0415
        from get_instance import GetData  # noqa: PLC0415
        root = Path(__file__).resolve().parents[4] / "local" / "runs" / "regime_banks"
        seals = dict(line.split(" sha256 ") for line in (root / "bank_seal.sha256").read_text(encoding="utf-8").splitlines() if line)
        gd = GetData()
        out = {}
        for name in ("train_c100.json", "train_c500.json"):
            raw = (root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != seals.get(name):
                raise RuntimeError(f"regime bank {name} does not match its seal")
            bank = json.loads(raw)
            inst = {k: {"capacity": int(v["capacity"]), "num_items": int(v["num_items"]), "items": [int(x) for x in v["items"]]}
                    for k, v in bank["instances"].items()}
            out[name[:-5]] = {"capacity": int(bank["capacity"]), "instances": inst,
                              "lb": float(gd.l1_bound_dataset(inst)), "sha256": seals[name]}
        return out

    def _regime_raw(self, code: str) -> "dict | None":
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        comp = {}
        for name, b in self._regime.items():
            res = run_sandboxed({"op": "energy", "code": code, "capacity": b["capacity"],
                                 "instances": {name: b["instances"]}, "lb": {name: b["lb"]},
                                 **({"eval_seed": self.eval_seed + 2} if self.eval_seed is not None else {})},
                                timeout=self.eval_timeout)
            if res is None or not res.get("ok"):
                return None
            raw = res.get("value")
            if raw is None or not np.isfinite(raw) or raw < 0:
                return None
            comp[name] = float(raw)
        return comp

    def energy(self, genes: dict) -> "float | None":
        if self.deterministic_only and uses_randomness(genes["code"]):
            self.randomness_rejects += 1
            return None  # lethal: non-deterministic scoring function (determinism gate)
        if self.objective == "regime_max":
            comp = self._regime_raw(genes["code"])
            if comp is None:
                return None
            import hashlib  # noqa: PLC0415
            self.last_components[hashlib.sha256(genes["code"].encode("utf-8")).hexdigest()] = comp
            return float(min(max(comp.values()), RAW_MAX) / RAW_MAX)
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        inst, lb = self._instances_payload()
        res = run_sandboxed({"op": "energy", "code": genes["code"],
                             "capacity": 100, "instances": inst, "lb": lb,
                             **({"eval_seed": self.eval_seed} if self.eval_seed is not None else {}),
                             **({"signature": True} if self.signature_source == "energy" else {})},
                            timeout=self.eval_timeout)
        if res is None or not res.get("ok"):
            return None
        if self.signature_source == "energy":
            self._store_signature(genes["code"], res.get("signature"))
        raw = res.get("value")
        if raw is None or not np.isfinite(raw) or raw < 0:
            return None
        return float(min(raw, RAW_MAX) / RAW_MAX)

    def endpoint_raw(self, genes: dict, bank: dict) -> "dict | None":
        """E4 endpoint: UNCLIPPED raw excess of one program PER INSTANCE of a
        sealed confirmation bank ({"capacity", "instances": {name: {capacity,
        num_items, items}}}), evaluated one instance per sandbox run so the
        per-instance values are replayable. Evaluation only;
        the bank never enters prompts. L1 lower bounds are computed on host
        (pure numpy, no candidate code). None if any instance fails."""
        from get_instance import GetData  # noqa: PLC0415
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        gd = GetData()
        per: dict = {}
        for name, v in bank["instances"].items():
            inst = {name: {"capacity": int(v["capacity"]), "num_items": int(v["num_items"]),
                           "items": [int(x) for x in v["items"]]}}
            lb = float(gd.l1_bound_dataset(inst))
            res = run_sandboxed({"op": "energy", "code": genes["code"],
                                 "capacity": int(bank["capacity"]),
                                 "instances": {"bank": inst}, "lb": {"bank": lb},
                             **({"eval_seed": self.eval_seed + 2} if self.eval_seed is not None else {})},

                                timeout=EVAL_TIMEOUT + 120)
            if res is None or not res.get("ok"):
                return None
            raw = res.get("value")
            if raw is None or not np.isfinite(raw):
                return None
            per[name] = float(raw)
        return per or None

    def _nl_matrix(self, genes: dict) -> "np.ndarray | None":
        import hashlib  # noqa: PLC0415
        text = " ".join(str(genes.get("thought_meas") or genes.get("thought") or "").split())  # R-meas: repaired text feeds the carrier only
        if not text:
            return None
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._keylock("embed", key):
            return self._nl_row_cached(key, text)

    def _nl_row_cached(self, key: str, text: str):
        v = self._embed_cache.get(key)
        if v is None:
            if self._embedder is None:
                from grant_evo.tgade.embeddings import OpenAIEmbedder  # noqa: PLC0415
                self._embedder = OpenAIEmbedder(model=NL_EMBED_MODEL)
            v = np.asarray(self._embedder.embed([text]), dtype=float).reshape(-1)
            n = float(np.linalg.norm(v))
            if v.shape[0] != NL_EMBED_DIM or not np.isfinite(n) or n < 1e-9:
                return None
            v = v / n
            self._embed_cache[key] = v
        return v.reshape(1, -1)

    @staticmethod
    def normalise_code(code: str) -> str:
        """Comment/docstring-free, ast-unparsed code (same pre-processing as the qualification replay)."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)

    def _store_signature(self, code: str, feat) -> None:
        """By-product signature -> L2 unit row (float32), cached by code sha; bounded cache
        (the engine only needs the current pool)."""
        import hashlib  # noqa: PLC0415
        ck = hashlib.sha256(code.encode("utf-8")).hexdigest()
        row = None
        if feat:
            v = np.asarray(feat, dtype=float).reshape(-1)
            n = float(np.linalg.norm(v))
            if v.shape[0] == 2 * len(self._probe) and np.isfinite(n) and n >= 1e-9:
                row = (v / n).astype(np.float32)
        with self._keylock("sig", "cache"):
            self._sig_cache[ck] = row
            while len(self._sig_cache) > 256:
                self._sig_cache.pop(next(iter(self._sig_cache)))

    def _behaviour_row(self, genes: dict) -> "np.ndarray | None":
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        if self.signature_source == "energy":
            import hashlib  # noqa: PLC0415
            ck = hashlib.sha256(genes["code"].encode("utf-8")).hexdigest()
            if ck not in self._sig_cache:
                self.energy(genes)  # one evaluation fills the cache (scripts / out-of-order callers)
            v = self._sig_cache.get(ck)
            return None if v is None else v.astype(float)
        res = run_sandboxed({"op": "signature", "code": genes["code"],
                             "capacity": 100,
                             "items": [int(x) for x in self._probe],
                             **({"eval_seed": self.eval_seed + 1} if self.eval_seed is not None else {})},
                            timeout=min(90.0, self.eval_timeout))
        if res is None or not res.get("ok"):
            return None
        v = np.asarray(res["value"], dtype=float).reshape(-1)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-9:
            return None
        return v / n

    def _hybrid_matrix(self, genes: dict) -> "np.ndarray | None":
        wb, wd, wc = self.hybrid_weights
        parts = []
        if wb > 0:
            b = self._behaviour_row(genes)
            if b is None:
                return None
            parts.append(np.sqrt(wb) * b)
        if wd > 0:
            d = self._nl_matrix(genes)
            if d is None:
                return None
            parts.append(np.sqrt(wd) * d.reshape(-1))
        if wc > 0:
            c = self._nl_matrix({"thought": self.normalise_code(genes["code"]), "code": genes["code"]})
            if c is None:
                return None
            parts.append(np.sqrt(wc) * c.reshape(-1))
        v = np.concatenate(parts)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-9:
            return None
        return (v / n).reshape(1, -1)

    def diversity_matrix(self, genes: dict) -> "np.ndarray | None":
        if self.diversity_carrier == "nl":
            return self._nl_matrix(genes)
        if self.diversity_carrier == "hybrid":
            return self._hybrid_matrix(genes)
        v = self._behaviour_row(genes)  # unit row
        return None if v is None else v.reshape(1, -1)
