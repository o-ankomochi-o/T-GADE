#!/usr/bin/env bash
# Launch one run of the arXiv v2 experiments against an OpenAI-compatible endpoint (the paper used gpt-oss-20b on a
# self-hosted vLLM server). Every argument below is the one used for the reported cohorts; only the endpoint, the
# model id and the output directory are yours.
#
#   export TGADE_LLM_BASE_URL=http://127.0.0.1:18000/v1     # OpenAI-compatible server; zero prices require this variable
#   bash experiments/hox_20261008/run_cohort.sh <family> <temperature> <client_seed> <out_dir> [model]
#
#   family       steady | steady_eoh | generational | speed_eoh | speed_host
#   temperature  selection temperature T (ignored for steady_eoh and speed_eoh)
#   client_seed  84001-84010 (steady family), 85001-85010 (generational), 86001-86010 (speed confirmation)
#
# steady        : steady-state Fermi-type (one individual per objective level) + host improve step, 1600 operator calls
# steady_eoh    : EoH reference (its own survival rule, no host), 3200 operator calls
# generational  : generational Bose-type + host improve step, 177 generations x (9 operator + 9 host) calls
# speed_eoh     : confirmation cohort arm A, EoH with 800 operator calls
# speed_host    : confirmation cohort arms B (T=0) and C (T=0.003), 400 operator calls + at most 400 host calls
set -euo pipefail
FAMILY=${1:?family}; T=${2:?temperature}; CS=${3:?client seed}; OUT=${4:?output dir}; MODEL=${5:-openai/gpt-oss-20b}
cd "$(dirname "$0")/../.."
: "${TGADE_LLM_BASE_URL:?set TGADE_LLM_BASE_URL to the OpenAI-compatible endpoint}"
PY=${PYTHON:-python}
BASE=(--confirm-paid --task bp_online --deterministic-only --eval-timeout 30 --seed 101 --cap-usd 0.01 --samplers 2 --evaluators 2
      --survival eoh --max-tokens 2048 --model "$MODEL" --price-in 0 --price-out 0)
THERMO=(--survival thermo --forced-template none --carrier behaviour --signature-source energy --exclusion level)
GEN=(--confirm-paid --task bp_online --model "$MODEL" --price-in 0 --price-out 0 --max-tokens 2048 --host-max-tokens 8192
     --occupancy boson --diversity-carrier behaviour --probe-len 64 --signature-source energy --deterministic-only --eval-timeout 30
     --mutation-template mix --n 8 --seed 101 --eval-workers 4 --cap-usd 0.01 --llm-workers 4 --operator-policy eoh
     --child-post-ops none --integrity full --integrity-kind improve --generations 177)
case "$FAMILY" in
  steady_eoh)   $PY scripts/run_v101_eoh.py --label "eoh3200_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 3200 --out "$OUT" ;;
  steady)       if [ "$T" = "0" ]; then
                  $PY scripts/run_v101_eoh.py --label "h_improve_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 1600 --host-repair improve --out "$OUT"
                else
                  $PY scripts/run_v101_eoh.py --label "h_improve_T${T}_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 1600 "${THERMO[@]}" --temperature "$T" --host-repair improve --out "$OUT"
                fi ;;
  generational) $PY scripts/run_v101.py --label "b_T${T}_cs$CS" --client-seed "$CS" --temperature "$T" "${GEN[@]}" --out "$OUT" ;;
  speed_eoh)    $PY scripts/run_v101_eoh.py --label "eoh800_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 800 --out "$OUT" ;;
  speed_host)   if [ "$T" = "0" ]; then
                  $PY scripts/run_v101_eoh.py --label "h_improve_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 400 --host-repair improve --out "$OUT"
                else
                  $PY scripts/run_v101_eoh.py --label "h_improve_T${T}_cs$CS" --client-seed "$CS" "${BASE[@]}" --calls 400 "${THERMO[@]}" --temperature "$T" --host-repair improve --out "$OUT"
                fi ;;
  *) echo "unknown family $FAMILY" >&2; exit 2 ;;
esac
