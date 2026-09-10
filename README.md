# T-GADE: Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts

Code and run records for the paper *T-GADE: Thermodynamical Generative-AI-Driven Evolution of LLM Artifacts*
(Ogawa and Mori, arXiv, 2026; identifier to be added).

T-GADE selects populations of LLM-generated artifacts by minimising a population free energy
`F_T(P) = <E>_P - T H(P)`, where `<E>_P` is the mean evaluator-derived energy and `H(P)` a per-individual
diversity measure, `H(P) = (1/|P|) sum_k log det(G_k(P) + eps I)` (log-volume of the feature Gram matrix).
The code minimises the total `|P| F_T(P) = sum E - T sum_k log det`, which has the same minimiser because every
selection step compares sets of equal size. Temperature `T` weights the diversity term against quality
in a deterministic minimisation. Occupancy is Fermi-type (one individual per genotype, i.e. per exact
description-and-code pair) or Bose-type (unrestricted, copies of one candidate allowed). The released
steady-state configuration labelled `L_level_T1e-5` uses the EoH rule (one individual per objective value),
not Fermi-type occupancy; see the configuration table below. The update schedule is steady-state (one individual replaced at a time) or generational.
The reference survival rule of Evolution of Heuristics (EoH) is recovered at T = 0 by steady-state selection
with one individual per objective value, under the common filtering and tie-breaking rules stated in the paper.

## Layout

| Path | Content |
|---|---|
| `src/grant_evo/tgade/` | free energy, thermodynamical selection, engine (generational update) |
| `src/grant_evo/bench/` | benchmark harness: online bin packing adapter, sandboxed evaluation, OpenRouter client, steady-state removal rule (`eoh_thermo.py`) |
| `scripts/run_v101.py` | generational T-GADE run (one configuration, one client seed) |
| `scripts/run_v101_eoh.py` | steady-state runs on the EoH loop: EoH reference, T-GADE removal rule, and the B1/B2 baselines |
| `scripts/analysis/` | aggregation scripts that run on the released records; `reference/` holds the scripts that need the full raw records |
| `data/gen0_bank.json` | sealed initial bank of eight artifacts (seed 101) shared by every configuration |
| `experiments/e4/confirmation_bank*.json` | capacity-100 and capacity-500 evaluation instances |
| `third_party/EoH/` | EoH at commit `4725457` (MIT): `eoh/src` and the `bp_online` problem definition |
| `results/runs/v101/` | per-run records of the 140 runs reported in the paper. Steady-state runs: `design.json`, `seeds.json`, `summary.json`, `selected.json`, `final.json`, `pop_endpoint.json`. Generational runs: `design.json`, `final.json`, `g1_history_best.json`, `pop_endpoint.json` (the generational runner writes no `summary.json`; its call and cost accounting is in `final.json`) |
| `results/paper_v1/` | `run_outcomes_v1.csv` (one row per run), per-run best-so-far curves, per-run operator outcome counts, and the selection-curve data behind the figures and tables |

The package directory keeps its historical name `grant_evo`.

## Setup

Python 3.11 or newer and Docker (candidate programs are evaluated in a `python:3.11-slim`-based container).
Build the evaluation image once before the first run that uses Docker:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from grant_evo.bench.sandbox import ensure_image; ensure_image()"
```

```bash
pip install -r requirements.txt
cp .env.example .env   # put your OpenRouter key in .env (never commit it)
```

The scripts add `src/` to `sys.path` themselves; no installation of the package is required.

A no-cost smoke test of the generational loop with a mock evaluator and a scripted LLM (no Docker, no API key):

```bash
python scripts/run_v101.py --offline --label smoke --generations 2 --n 8 --seed 101 --client-seed 1 --eval-workers 1 --llm-workers 1 --deterministic-only --eval-timeout 30 --operator-policy eoh --integrity skip --child-post-ops none
```

## Reproducing one run

All paper runs use the model `qwen/qwen3-32b` through OpenRouter, the sealed bank, a 30 s cap on training
evaluation (confirmation and transfer evaluation: 180 s per instance), the determinism screen, and client seeds
20501-20510 and 20601-20610. Each of the reported runs cost about 0.15 USD at the time; prices are not guaranteed.

Steady-state EoH reference (800 calls):

```bash
python scripts/run_v101_eoh.py --confirm-paid --label main_eoh_800_evt30_s2e2_cs20601 --calls 800 --seed 101 \
  --client-seed 20601 --cap-usd 0.3 --eval-timeout 30 --deterministic-only --samplers 2 --evaluators 2 --survival eoh
```

Steady-state T-GADE removal rule (EoH rule = one individual per objective value, T = 1e-5; Bose-type: unrestricted, T = 0.003; `--exclusion genotype` gives Fermi-type occupancy):

```bash
python scripts/run_v101_eoh.py --confirm-paid --label x1_level_T0.00001_800_bh_sigE_s2e2_cs20601 --calls 800 --seed 101 \
  --client-seed 20601 --cap-usd 0.3 --eval-timeout 30 --deterministic-only --samplers 2 --evaluators 2 \
  --survival thermo --exclusion level --temperature 0.00001 --carrier behaviour --hybrid-weights 0.8,0.1,0.1 \
  --probe-len 64 --signature-source energy --forced-template m2
```

(use `--exclusion none --temperature 0.003` for the Bose-type configuration.)

Generational T-GADE (88 generations x 9 offspring = 792 calls; occupancy `boson` = Bose-type, unrestricted; `fermion` = Fermi-type, one individual per exact description-and-code pair):

```bash
python scripts/run_v101.py --confirm-paid --label g1_x2full_bh_sigE_boson_T0.003_evt30_cs20601 --temperature 0.003 \
  --occupancy boson --diversity-carrier behaviour --probe-len 64 --signature-source energy --deterministic-only \
  --eval-timeout 30 --mutation-template mix --n 8 --seed 101 --client-seed 20601 --eval-workers 4 --cap-usd 1.0 \
  --llm-workers 4 --child-post-ops none --integrity skip --operator-policy eoh --generations 88
```

Baselines: B1 (parent-free generation) adds `--survival eoh --operators i1`; B2 (quality-blind evolution) adds
`--survival newest --parent-selection uniform` to the steady-state command.

Outputs go to `runs/`. The client seed fixes the per-call seed schedule sent to the provider; provider responses are not guaranteed to be reproducible.

## Reproducing the tables and figures from the released records

| Output | Command / file |
|---|---|
| Training comparison (Table II) | `python scripts/analysis/sixarm_pooled_board.py` (reads `summary.json` for steady-state runs, `final.json` and `g1_history_best.json` for generational runs; prints generational configurations as `G1-...` and steady-state ones as `R-...`, see the name map below) |
| Deployment curves and population composition (Fig. 4, Table IV) | `python scripts/analysis/topk_sixarm_pooled.py` (reads `pop_endpoint.json`; writes to `results/paper_v1/`; its preview plot uses a log-scale y axis, the paper figure a linear axis) |
| Main comparison statistics (Mann-Whitney, Cliff's delta, <= 1.0% attainment, baseline medians) | `python scripts/analysis/main_comparison_stats.py` (reads `run_outcomes_v1.csv`) |
| Selection activity per run (Fig. 2b) | `results/paper_v1/selection_activity_v1.csv` (per steady-state T-GADE run: logged removals and the fraction that discard a lower recorded objective while a higher one survives) |
| Best-so-far curves (Fig. 3) | `results/paper_v1/per_run_best_so_far_20260909.json` (one curve per run, 9-call grid) and its medians/quartiles in `fig2_best_so_far_20260909.json` |
| Operator outcomes (Table III) | `results/paper_v1/per_run_operator_outcomes_20260909.json` (per run and operator: better / equal / worse counts) and the pooled `table6_operator_outcomes_20260909.json` |
| One row per run (Appendix table, Fig. 2a) | `results/paper_v1/run_outcomes_v1.csv` |

The per-run JSON files were produced from the complete run records by `scripts/analysis/reference/make_fig2_table6.py`.
The complete candidate-level records are not part of this release. `scripts/analysis/reference/` documents the population re-evaluation (`pop_c500*.py`) and
history-best re-scoring (`g1_history_best*.py`) that produced `pop_endpoint.json` and `g1_history_best.json`.

## Conventions moved out of the paper

The paper states the experimental settings in words; the exact bookkeeping lives here.

**Serving and sampling.** All main runs call `qwen/qwen3-32b` through OpenRouter with the serving provider pinned
to SiliconFlow and provider fallbacks disabled (a different serving provider can change the outputs of the same model).
Sampling temperature 0.8, at most 2,048 output tokens, prompts prefixed with `/no_think`. The thermodynamical
selection temperature `T` is a different parameter. Candidate programs run in an isolated container; training
evaluation has a 30 s timeout, confirmation and transfer evaluation 180 s per instance. Programs that use
random-number APIs are rejected before evaluation.

**Endpoints and precision.** For generational runs the reported endpoint is the best training artifact in the final
population (`final.json`, field `train_raw_excess_best`); for steady-state runs it is the history-best artifact
(`summary.json`, field `train_raw_excess_best_unclipped`). Summaries use these unrounded values. For rank statistics
both groups are rounded to five decimal places, the precision of the reference population manager; in the released
data this changes no rank or tie. Training excess is a ratio of sums over the five capacity-100 training
instances, `(sum of bins used - sum of lower bounds) / sum of lower bounds`, as computed by the EoH evaluator;
confirmation and transfer excess is the mean over instances of the per-instance ratio
`(bins used - lower bound) / lower bound`. Selection uses `E = min(excess, 2) / 2`.

**Diversity features.** The behaviour signature of an artifact is taken on the first training instance: for each
of its 5,000 packing decisions, the fill ratio of the chosen bin and the tightness rank of that bin, concatenated
into a 10,000-dimensional vector and L2-normalised. The Gram regulariser is `eps = 1e-3` (confirmed in the saved run
configurations). With `--signature-source energy` the full 5,000-decision signature is used and `--probe-len` has
no effect on it.

**Deployment selection (P1, P2, P3).** Each final population is ranked by training objective. P2 takes the top-k
artifacts, selects one on the first capacity-500 instance (`conf500_90101`), and reports the mean excess on the
remaining four instances; P1 takes the training-best artifact and reports the same four; P3 selects and scores on
all five instances (diagnostic; selection and test are not separated). An artifact whose evaluation fails or times
out receives the worst attainable excess of that bank; a missing per-instance value falls back to the aggregate.

**Call accounting.** Budgets count logical generation calls: 800 for steady-state runs, 88 x 9 = 792 for
generational runs. Transport retries are recorded separately and not counted. Re-evaluation of retained parents
and all deployment scoring use the evaluator only and consume no generation call.

**Seeds.** Client seeds 20501-20510 and 20601-20610 for the six main configurations; 20601-20610 for the two
baselines. The client seed fixes the per-call seed schedule sent to the provider; provider responses are not
guaranteed to be reproducible.

**EoH reference.** The steady-state EoH runs use the EoH code at commit `472545785c936dcfc863d2bc0d6109cf23c7ce62`
(vendored under `third_party/EoH`) with the same evaluator, initial bank and call ledger as the T-GADE runs.
They are a reference comparison on the same task under the model, population size (8), parents per operator (2),
operator set (E1, E2, M1, M2) and call budget shared by every configuration here; they are not a reproduction of
the settings of the 2024 EoH paper (GPT-3.5, population 20, 20 generations, five parents, five operators, about
2,000 calls).

**Configuration names.** The paper names each configuration by method, update schedule, occupancy and
temperature. The released records keep their original labels: the CSV `arm` column, the run-directory prefixes and
the keys inside the per-run and pooled JSON files are unchanged, and the analysis scripts read those labels. This
table is the display-name map.

| Paper name (method, update, occupancy, T) | CSV `arm` (internal id) | Label in per-run JSON (internal id) | Label in pooled deployment JSON (internal id) | Run-directory prefix (internal id) |
|---|---|---|---|---|
| EoH, steady-state, objective-value deduplication, - | `EoH` | `EoH` | `EoH (steady)` | `eoh_main_eoh_800_evt30_s2e2_cs` |
| T-GADE, steady-state, one per objective value (EoH rule), 1e-5 | `L_level_T1e-5` | `L (T=1e-5, level)` | `L 1e-5 (steady, EoH replicate)` | `eoh_x1_level_T0.00001_800_bh_sigE_s2e2_cs` |
| T-GADE, steady-state, Bose-type, 0.003 | `N_none_T0.003` | `N (T=0.003, unrestricted)` | `N 0.003 (steady)` | `eoh_x1_none_T0.003_800_bh_sigE_s2e2_cs` |
| T-GADE, generational, Fermi-type (one per genotype), 0 | `G_fermion_T0` | `Gen. fermion T=0` | `G1 fermion T=0 (gen.)` | `g1_x2full_bh_sigE_fermion_T0_evt30_cs` |
| T-GADE, generational, Bose-type, 0.003 | `G_boson_T0.003` | `Gen. boson T=0.003` | `G1 boson 0.003 (gen.)` | `g1_x2full_bh_sigE_boson_T0.003_evt30_cs` |
| T-GADE, generational, Bose-type, 0.03 | `G_boson_T0.03` | `Gen. boson T=0.03` | `G1 boson 0.03 (gen.)` | `g1_x2full_bh_sigE_boson_T0.03_evt30_cs` |
| independent generation (baseline) | `B1_parent_free` | `B1 parent-free` | - | `eoh_b1_i1only_800_evt30_s2e2_cs` |
| no quality-based selection (baseline) | `B2_quality_blind` | `B2 quality-blind` | - | `eoh_b2_newest_uniform_800_evt30_s2e2_cs` |

The pooled operator-outcome file joins the per-run JSON label with the operator name (for example
`Gen. boson T=0.003|eoh_e1`); the per-run operator file is keyed by run directory.

## License

MIT (see `LICENSE`). `third_party/EoH` is the EoH code at commit `4725457` and keeps its own MIT license
and copyright notice (`third_party/EoH/LICENSE`).

## Citation

To be added once the arXiv identifier is assigned.
