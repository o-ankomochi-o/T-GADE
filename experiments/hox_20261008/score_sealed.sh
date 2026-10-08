#!/usr/bin/env bash
# Score finished runs on a sealed bank (CPU only).
#   bash experiments/hox_20261008/score_sealed.sh <steady|generational> <cohort_dir> <out.json> [c100|c500]
# <cohort_dir> holds one run directory per arm and seed as written by run_cohort.sh.
set -euo pipefail
FAMILY=${1:?steady|generational}; COHORT=${2:?cohort dir}; OUT=${3:?output json}; BANK=${4:-c100}
cd "$(dirname "$0")/../.."
S=experiments/hox_20261008; PY=${PYTHON:-python}
case "$FAMILY" in
  steady)       $PY $S/scorers/hoxN_eval_sealed.py --bank $S/sealed_bank/sealed_$BANK.json --seal $S/sealed_bank/seal.sha256 --out "$OUT" "$COHORT" ;;
  generational) $PY $S/scorers/hox5_eval_sealed_budgets.py --bank $S/sealed_bank/sealed_$BANK.json --seal $S/sealed_bank/seal.sha256 --out "$OUT" "$COHORT" ;;
  *) echo "unknown family $FAMILY" >&2; exit 2 ;;
esac
