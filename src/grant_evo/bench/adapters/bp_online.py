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
                 signature_source: str = "probe", deterministic_only: bool = False,
                 noise_sigma: float = 5.0, noise_kappa: float = 5.0, noise_mode: str = "sto",
                 noise_seed_base: int = 900000):
        EV, EoHConfig, LLMConfig, BPONLINE = _lazy_imports()
        # NOISY prereg (2026-10-01): objective "noisy_c100" = noisy-observation online bin packing on the
        # train items (sandbox op energy_noisy). noise_mode "sto" draws a new noise seed per evaluation
        # (noise_seed_base + evaluation counter); "det" fixes noise_seed_base. Seeds are recorded per evaluation.
        if noise_mode not in ("sto", "det"):
            raise ValueError(f"unknown noise_mode: {noise_mode!r}")
        self.noise_sigma, self.noise_kappa, self.noise_mode = float(noise_sigma), float(noise_kappa), noise_mode
        self.noise_seed_base, self._noise_counter = int(noise_seed_base), 0
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
        if objective not in ("train_c100", "regime_max", "noisy_c100"):
            raise ValueError(f"unknown objective: {objective!r}")
        self.objective = objective
        if objective == "noisy_c100":
            self.energy_spec = (f"E = min(raw, {RAW_MAX}) / {RAW_MAX}; raw = mean over datasets of (mean J - L1 bound) / L1 bound, "
                                f"J = bins + {self.noise_kappa} * overflows, integer observation x = clip(round(w + N(0, {self.noise_sigma}^2)), 1, 100), "
                                f"noise_mode={noise_mode}, seed base {self.noise_seed_base}")
        # Step 6 integrity: the repair the host model must own is
        # description<->code coherence, not syntax. "syntax" = legacy syntax-only
        # code repair (thought untouched); "align_thought" = the individual's host
        # rewrites the description to match the code; "align_code" = the host makes
        # the code implement the description.
        # "align_thought_meas" (replan round 2026-09-06, R-meas): the repaired
        # description is stored in genes["thought_meas"] and feeds ONLY the NL
        # diversity carrier; genes["thought"] (what operators see) and the
        # loci digest stay untouched. Repairs are cached by code sha.
        if integrity_kind not in ("syntax", "align_thought", "align_thought_meas", "align_code", "align_thought_eg", "align_code_eg",
                                  "improve", "improve_nocard", "oracle", "oracle_inject"):
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
        if signature_source == "energy" and objective not in ("train_c100", "noisy_c100"):
            raise ValueError("signature_source='energy' needs the train_c100 or noisy_c100 objective")
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
        prompt = getattr(self, "ALIGN_THOUGHT_PROMPT_TASK", ALIGN_THOUGHT_PROMPT) + genes["code"]  # task override (TSP)
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

    # ── execution-grounded HoxLM: the host sees what the code actually does ──
    EXEC_PROBE_LEN = 64
    INTERFACE_SPEC = ("def score(item: int, bins: np.ndarray) -> np.ndarray; `bins` holds the remaining capacities of the "
                      "feasible bins (all >= item); return one float score per bin (same length as bins); the bin with the "
                      "highest score receives the item; numpy is available as np; no randomness")

    def _exec_probe(self, code: str) -> dict:
        """Run the candidate on a fixed 64-item probe in the sandbox (never on the host; the host only compiles it).
        Returns {"ok": bool, "summary": str} for the host prompt and for the keep-original rule."""
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            return {"ok": False, "summary": f"the code does not compile: {exc.msg} (line {exc.lineno})"}
        if self.deterministic_only and uses_randomness(code):
            return {"ok": False, "summary": "the code calls a randomness API, which the evaluator rejects"}
        items = [int(x) for x in list(self._probe)[: self.EXEC_PROBE_LEN]]
        res = run_sandboxed({"op": "signature", "code": code, "capacity": 100, "items": items, "ties": True},
                            timeout=min(90.0, self.eval_timeout))
        if res is None or not res.get("ok"):
            err = (res or {}).get("error") or "timeout or abnormal exit"
            what = ("the function could not be defined (import or definition error)" if "initialise" in err else
                    "calling score(item, bins) failed: it raised an exception or did not return one score per feasible bin")
            return {"ok": False, "summary": f"{what} on a {len(items)}-item probe"}
        feat = [float(x) for x in res["value"]]
        fills, ranks = feat[0::2], feat[1::2]
        n = max(1, len(ranks))
        tie = res.get("ties", 0) / max(1, res.get("choices", 0))  # share of the steps with a real choice
        if res.get("choices", 0) and tie >= 0.99:  # a constant score expresses no preference; say only what decided the placement
            return {"ok": True, "tie_share": tie, "summary": (
                f"it runs on a {len(ranks)}-item probe, but its score was the same for every feasible bin at almost every "
                "step with a real choice, so those placements were decided by tie-breaking (the first feasible bin in bin order), not by the score")}
        tight = sum(1 for r in ranks if r <= 1e-9) / n
        new_bin = sum(1 for f, it in zip(fills, items) if abs(f - round(it / 100.0, 4)) < 1e-6) / n
        tie_txt = (f"; in {tie:.0%} of the steps with bins of different remaining capacity, several of them shared the top score and the placement was decided by "
                   "tie-breaking (the first such bin in bin order)") if tie > 0 else ""
        return {"ok": True, "tie_share": tie, "summary": (
            f"it runs on a {len(ranks)}-item probe; observed placements: the tightest feasible bin in {tight:.0%} of steps, "
            f"an empty bin in {new_bin:.0%}; mean fill after placement {sum(fills) / n:.2f}{tie_txt}")}

    def _eg_record(self, row: dict) -> None:
        """Minimal EG evidence with candidate identity (the eg_log is persisted, never in-memory only)."""
        self.eg_log = getattr(self, "eg_log", [])
        self.eg_log.append(row)
        path = getattr(self, "eg_log_path", None)
        if path:
            with self._keylock("eg", "log"), open(path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _integrity_eg(self, genes: dict, llm) -> dict:
        """align_thought_eg: redescribe the code given its measured behaviour (code kept).
        align_code_eg: repair the code given the interface and execution feedback; keep the ORIGINAL if the repair fails
        the probe or returns no code (an invalid repair never removes a child; a valid repair can still score worse,
        which only the training evaluations of original and repair show)."""
        import re as _re  # noqa: PLC0415
        from grant_evo.tgade.engine import loci_canonical_digest as _dig  # noqa: PLC0415
        probe = self._exec_probe(genes["code"])
        row = {"kind": self.integrity_kind, "pre_digest": _dig(self.loci_view(genes)), "probe_ok": probe["ok"],
               "tie_share": probe.get("tie_share"), "probe": probe["summary"]}
        if self.integrity_kind == "align_thought_eg":
            base = getattr(self, "ALIGN_THOUGHT_PROMPT_TASK", ALIGN_THOUGHT_PROMPT)
            prompt = (base.rstrip() + f"\nMeasured behaviour from running the code: {probe['summary']}. The sentence must be "
                      "consistent with this measured behaviour.\n\n" + genes["code"])
            out = llm(prompt, temperature=0.0, call_id=f"repair-eg-thought-{hashlib.sha256(genes['code'].encode('utf-8')).hexdigest()[:32]}")
            text = out if isinstance(out, str) else (getattr(out, "text", None) or "")
            m = _re.search(r"\{(.*?)\}", text or "", _re.S)
            new_t = " ".join(m.group(1).split()) if (m and m.group(1).strip()) else None
            out_g = {"thought": new_t if new_t is not None else genes["thought"], "code": genes["code"]}
            self._eg_record({**row, "result": "redescribed" if new_t is not None else "no_sentence_kept_original",
                             "post_digest": _dig(self.loci_view(out_g))})
            return out_g
        rule = getattr(self, "ALIGN_CODE_RULE", "scoring rule")
        prompt = (f"/no_think\nThe description below states the intended {rule}; the code must implement exactly that {rule} "
                  f"and must run. Interface: {self.INTERFACE_SPEC}. Execution feedback for the current code: {probe['summary']}. "
                  "If the code already implements the description and runs, return it unchanged; otherwise minimally change it "
                  "so that it does. Keep the function name, inputs and outputs; return the full function in a ```python code block.\n\n"
                  f"Description: {genes['thought']}\n\nCode:\n{genes['code']}")
        out = llm(prompt)
        text = out if isinstance(out, str) else (getattr(out, "text", None) or "")
        _, code = self.evo._extract(text)
        if not code:
            self._eg_record({**row, "result": "no_code_kept_original", "post_digest": row["pre_digest"]})
            return dict(genes)
        new_code = self.evo._prepend_imports(code[0])
        after = self._exec_probe(new_code)
        ok_valid, _ = self.validate({"thought": genes["thought"], "code": new_code})
        if not (after["ok"] and ok_valid):
            self._eg_record({**row, "result": "repair_failed_kept_original", "post_digest": row["pre_digest"],
                             "repair_probe": after["summary"]})
            return dict(genes)
        out_g = {"thought": genes["thought"], "code": new_code}
        self._eg_record({**row, "result": "repaired" if new_code != genes["code"] else "unchanged",
                         "post_digest": _dig(self.loci_view(out_g)), "repair_probe": after["summary"]})
        return out_g

    # ── HoxLM "improve" host ─────────────────────────────
    # The host is the ONLY component that sees measurement: the child's training score, its execution probe and the
    # parents' scores (EoH operators see descriptions and code only). keep-best: the repaired code is evaluated and
    # replaces the child only when it scores strictly better, so on the training objective the host never harms.
    DOMAIN_CARD_BP = (
        "Domain knowledge used by strong online bin packing heuristics:\n"
        "1. Best fit (prefer the feasible bin with the SMALLEST remaining capacity after placement, i.e. smallest bins - item) "
        "is the baseline; strong heuristics are best fit plus corrections. Scores that grow with the remaining capacity "
        "(worst fit) or that are the same for every bin (then the first bin is always taken) are weak.\n"
        "2. What matters is the residual capacity AFTER placement (bins - item), not bins itself. Residual 0 (exact fit) is "
        "ideal and deserves a large bonus. A small positive residual that no future item can use is wasted space: penalise "
        "residuals that are below the smallest common item size. Residuals large enough to hold a typical item are fine.\n"
        "3. Training instances: capacity 100, item sizes are integers 1-94 with median 39, quartiles 29-50 and 5th-95th "
        "percentile 16-65 (Weibull-shaped). Hence residuals below about 15 are almost never reusable, residuals of 30-50 are "
        "easy to reuse, and an empty bin (residual 100 - item) is reusable but opening it costs a bin.\n"
        "4. Do not open an empty bin while a partially filled bin fits the item, unless every partially filled bin would be "
        "left with an unusable residual.\n"
        "5. Index preferences (earlier bins first) only as a tie-breaker with a tiny weight; large index terms overfit the "
        "instance size and transfer badly to other capacities.\n"
        "6. The function must return one float per bin, deterministic, vectorised numpy, no randomness, no NaN or inf "
        "(add a small epsilon before dividing).")

    def host_repair(self, genes: dict, parents: "list | None", llm, mode: "str | None" = None,
                    fallback: "dict | None" = None) -> "tuple[dict, dict]":
        """kinds improve / improve_nocard / oracle: measurement-driven development of the child with keep-best on the
        training objective; oracle_inject: no LLM, returns the reference
        candidate for the first three children (pipeline positive control); other kinds -> integrity().
        mode="mask": the child is thought(parent A) + code(parent B) and the host reconstructs one coherent child; the
        runner passes the intact parent B as `fallback`, so an unreconstructed hybrid never enters the population (
        review 01:08). A proposal with new code but no braced description is not accepted either (joint update rule).
        Returns (genes, info); the improve kinds never return None genes (keep-best falls back to `fallback` or the child)."""
        kind = self.integrity_kind
        fb = dict(fallback) if fallback else dict(genes)
        if kind == "oracle_inject":
            with self._keylock("host", "inject"):
                self._inject_n = getattr(self, "_inject_n", 0) + 1
                n = self._inject_n
            if n <= 3 and getattr(self, "oracle_ref", None):
                return ({"thought": self.oracle_ref["thought"], "code": self.oracle_ref["code"]},
                        {"kind": kind, "result": "injected", "inject_index": n})
            return dict(genes), {"kind": kind, "result": "passthrough", "inject_index": n}
        if kind not in ("improve", "improve_nocard", "oracle"):
            out = self.integrity(genes, random.Random(0), llm)
            return out, {"kind": kind}
        pct = lambda e: None if e is None else round(200.0 * e, 3)  # energy = raw/2 -> raw excess in %  # noqa: E731
        pre_e = self.energy(genes)
        probe = self._exec_probe(genes["code"])
        par = [p for p in (parents or []) if isinstance(p, dict) and p.get("code")]
        scored = [p for p in par if p.get("objective") is not None]
        best_par = min(scored, key=lambda p: p["objective"]) if scored else None
        task = getattr(self.problem, "task_description", "") or ""
        meas = (f"training excess over the lower bound = {pct(pre_e)} % (lower is better; best fit scores 3.984 %)"
                if pre_e is not None else "it FAILED on the training instances (exception, timeout, invalid output or randomness)")
        parts = ["/no_think", "You develop a candidate heuristic inside an evolutionary search for online bin packing. " + task.strip(),
                 f"Interface: {self.INTERFACE_SPEC}."]
        if kind != "improve_nocard":
            parts.append(self.DOMAIN_CARD_BP + "\nThis knowledge is a menu of options, not a recipe: do not turn every candidate into "
                         "plain best fit, and do not add exact-fit rewards, residual penalties and reservation of space all at once.")
        if kind == "oracle" and getattr(self, "oracle_ref", None):
            ref = self.oracle_ref
            parts.append(f"Reference heuristic known to score {round(100 * ref['train_raw_excess'], 3)} % on these training instances "
                         f"(you may use any of its ideas or adopt it entirely):\n```python\n{ref['code']}\n```")
        if mode == "mask":
            parts.append("The candidate below was assembled from two parents: its description comes from parent A and its code from "
                         "parent B, so they do not match yet. Reconstruct ONE coherent candidate that keeps the strongest idea of each.")
        else:
            parts.append("The candidate below was produced by a crossover or mutation step.")
        parts.append(f"Measured on the TRAINING instances: {meas}. Execution probe: {probe['summary']}.")
        if best_par is not None:
            parts.append(f"Its best parent scores {round(100 * float(best_par['objective']), 3)} %:\n```python\n{best_par['code']}\n```")
        parts.append(f"Candidate description: {genes['thought']}\n\nCandidate code:\n```python\n{genes['code']}\n```")
        if kind == "oracle" and getattr(self, "oracle_ref", None):  # the generic one-change rule conflicted with adoption
            parts.append("Task: the reference heuristic above is known to be better on the training instances. Adopt it fully or "
                         "combine it with the candidate; full replacement is allowed and preferred when the candidate is weaker. Keep "
                         "the exact interface. Return one sentence describing the resulting algorithm inside braces, then the complete "
                         "function in one ```python code block.")
        else:
            parts.append("Task: preserve the candidate's useful principle and the exact interface. Repair any inconsistency between the "
                         "description and the code, and make ONE specific change that is likely to reduce the number of bins, justified by "
                         "the training evidence above (state which measured weakness it addresses). If the description itself encodes the "
                         "defect, update the description and the code together; matching a bad description is not the objective. Do not "
                         "copy the function template, and do not rewrite working code without a concrete reason. Return one sentence "
                         "describing the resulting algorithm inside braces, then the complete function in one ```python code block.")
        prompt = "\n\n".join(parts)
        info = {"kind": kind, "mode": mode, "pre_energy": pre_e, "pre_pct": pct(pre_e), "probe_ok": probe["ok"],
                "tie_share": probe.get("tie_share"),
                "best_parent_pct": None if best_par is None else round(100 * float(best_par["objective"]), 3)}
        out = llm(prompt)
        text = out if isinstance(out, str) else (getattr(out, "text", None) or "")
        alg, code = self.evo._extract(text)
        if not code:
            return fb, {**info, "result": "no_code_kept_fallback", "post_energy": None}
        new_code = self.evo._prepend_imports(code[0])
        unchanged = new_code.strip() == genes["code"].strip()  # the extractor strips the block: compare without trailing bytes
        if unchanged:
            new_code = genes["code"]  # keep the exact bytes (digest and cache stability)
        has_desc = bool(alg and alg[0].strip())
        new_thought = " ".join(alg[0].split())[:2000] if has_desc else genes["thought"]
        info.update(code_changed=not unchanged, thought_changed=new_thought != genes["thought"], has_description=has_desc)
        if not unchanged and not has_desc:  # new code must come with its description (joint update)
            return fb, {**info, "result": "no_description_kept_fallback", "post_energy": None}
        cand = {"thought": new_thought, "code": new_code}
        if unchanged:
            if has_desc and new_thought != genes["thought"]:  # description-only reconstruction: same bytes, same energy
                return cand, {**info, "result": "redescribed", "post_energy": pre_e, "post_pct": pct(pre_e)}
            return fb, {**info, "result": "unchanged_kept_fallback", "post_energy": pre_e, "post_pct": pct(pre_e)}
        ok_valid, _ = self.validate(cand)
        post_e = self.energy(cand) if ok_valid else None
        info.update(post_energy=post_e, post_pct=pct(post_e))
        if post_e is not None and (pre_e is None or post_e < pre_e):
            return cand, {**info, "result": "repaired_better"}
        return fb, {**info, "result": "repair_not_better_kept_fallback" if post_e is not None else "repair_invalid_kept_fallback"}

    def integrity(self, genes: dict, rng: random.Random,
                  llm: LLMClient) -> "dict | None":
        import re as _re  # noqa: PLC0415
        if self.integrity_kind in ("improve", "improve_nocard", "oracle", "oracle_inject"):
            return self.host_repair(genes, None, llm)[0]
        if self.integrity_kind in ("align_thought_eg", "align_code_eg"):
            return self._integrity_eg(genes, llm)
        if self.integrity_kind in ("align_thought", "align_thought_meas"):
            ck = hashlib.sha256(genes["code"].encode("utf-8")).hexdigest()
            with self._keylock("repair", ck):
                return self._align_thought_cached(ck, genes, llm)
        if self.integrity_kind == "align_code":
            rule = getattr(self, "ALIGN_CODE_RULE", "scoring rule")  # task override (TSP)
            prompt = (f"/no_think\nThe description below states the intended {rule}; the code must implement exactly that {rule}. "
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
        if self.objective == "noisy_c100":
            res = run_sandboxed(self.noisy_payload(genes["code"], self.next_noise_seed(),
                                                   signature=self.signature_source == "energy"), timeout=self.eval_timeout)
        else:
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

    # ── NOISY prereg (2026-10-01) ──────────
    def next_noise_seed(self) -> int:
        """sto: a new seed per evaluation (base + counter); det: the fixed base."""
        if self.noise_mode == "det":
            return self.noise_seed_base
        with self._keylock("noise", "counter"):
            self._noise_counter += 1
            return self.noise_seed_base + self._noise_counter

    def noisy_payload(self, code: str, noise_seed: int, signature: bool = False) -> dict:
        inst, lb = self._instances_payload()
        return {"op": "energy_noisy", "code": code, "capacity": 100, "instances": inst, "lb": lb,
                "sigma": self.noise_sigma, "kappa": self.noise_kappa, "noise_seed": int(noise_seed),
                **({"signature": True} if signature else {})}

    def noisy_panel(self, code: str, seeds: "list[int]", workers: int = 4) -> "dict | None":
        """Fixed-seed panel: raw (excess-loss ratio) and per-instance [B, O, J] for every seed; None if any fails."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        from grant_evo.bench.sandbox import run_sandboxed  # noqa: PLC0415
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            outs = list(ex.map(lambda s: run_sandboxed(self.noisy_payload(code, s), timeout=self.eval_timeout), seeds))
        if any(o is None or not o.get("ok") or o.get("value") is None for o in outs):
            return None
        raws = [float(o["value"]) for o in outs]
        return {"seeds": [int(s) for s in seeds], "raw": raws, "raw_mean": float(np.mean(raws)),
                "per_instance": [o.get("per_instance") for o in outs],
                "J_mean": float(np.mean([np.mean([v[2] for v in o["per_instance"].values()]) for o in outs])),
                "B_mean": float(np.mean([np.mean([v[0] for v in o["per_instance"].values()]) for o in outs])),
                "O_mean": float(np.mean([np.mean([v[1] for v in o["per_instance"].values()]) for o in outs]))}

    def noisy_prompt(self) -> "tuple[str, str]":
        """(task_description, template_program) shown to the generator for the noisy task; the
        evaluator contract (observed integer item, observed-x valid mask, empty bins, overflow rule) is stated."""
        s, k = self.noise_sigma, self.noise_kappa
        task = ("Design a novel score function that scores a set of bins to assign an item in ONLINE bin packing "
                "with NOISY item sizes. The size you see is an integer observation of the true size: "
                f"observed = clip(round(true + noise), 1, 100) with noise ~ Normal(0, sigma^2), sigma = {s:g} (known). "
                "In each step the item is assigned to the bin with the maximum score among bins whose remaining "
                "capacity is >= the observed size; bins with remaining capacity 100 are empty bins (choosing one "
                "opens a new bin; an empty bin never overflows). After placement the true size is applied: if it "
                "exceeds the bin's remaining capacity the bin OVERFLOWS, is closed for good, and a penalty of "
                f"{k:g} bins is added. The final goal is to minimize J = (number of used bins) + {k:g} * (number of overflows).")
        template = f'''
def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score each bin for assigning the current item. Higher score = preferred bin.

    Args:
        item: OBSERVED integer size of the current item (true size = item + noise, noise ~ Normal(0, {s:g}^2))
        bins: remaining capacities of bins with remaining capacity >= observed size (bins equal to 100 are empty)
    Returns:
        scores: priority scores for each bin
    """
    return bins
'''
        return task, template

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
