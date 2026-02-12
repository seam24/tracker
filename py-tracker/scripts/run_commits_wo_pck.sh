#!/bin/bash
#SBATCH --job-name=latentmu_commit_4b_scratch
#SBATCH --output=/dev/null #/scratch/users/trostami/commits/thefuck-4b/logs/commit_%A_%a.out
#SBATCH --error=/dev/null #/scratch/users/trostami/commits/thefuck-4b/logs/commit_%A_%a.err
#SBATCH --array=1-167
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

# Scratch base (repo + data + results live here)
BASE="/scratch/users/trostami/commits/thefuck-4b"
REPO="${BASE}/thefuck"
DATA_DIR="${BASE}/data"
LOGDIR="${BASE}/logs"

# Runner code still on GPFS (unless you copy it to scratch)
RUNNER="/mnt/aiongpfs/users/trostami/LatentMu/src/commit_test_runner.py"

RESULTS_BASE="${DATA_DIR}/program_results"
MUTANT_HISTORY="${DATA_DIR}/mutant_history.json"
COMMITS_FILE="${DATA_DIR}/commits_to_run.txt"

CONDA_ENV="thefuck-32b-env"
PYTEST_TIMEOUT="10"
TESTS_LIST=""
IGNORE_LIST=""
RUN_MODE="both"

# mkdir -p "$RESULTS_BASE" "$LOGDIR"

# Sanity checks (clear errors)
[[ -d "$REPO" ]] || { echo "Missing repo dir: $REPO" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 2; }
[[ -f "$COMMITS_FILE" ]] || { echo "Missing commits file: $COMMITS_FILE" >&2; exit 2; }
[[ -f "$MUTANT_HISTORY" ]] || { echo "Missing mutant history: $MUTANT_HISTORY" >&2; exit 2; }

source ~/miniconda3/etc/profile.d/conda.sh

LINE_NO="${SLURM_ARRAY_TASK_ID}"
COMMIT="$(sed -n "${LINE_NO}p" "$COMMITS_FILE" || true)"

if [[ -z "${COMMIT}" ]]; then
  echo "No commit found for line ${LINE_NO}, exiting."
  exit 0
fi

HEAD_COMMIT="$(head -n1 "$COMMITS_FILE")"
EXTRA_HEAD=()
if [[ "$COMMIT" == "$HEAD_COMMIT" ]]; then
  EXTRA_HEAD=(--use-head)
fi

# Per-commit results
OUTDIR="${RESULTS_BASE}"
mkdir -p "$OUTDIR"

TIMEOUT_ARG=()
if [[ -n "${PYTEST_TIMEOUT}" ]]; then
  TIMEOUT_ARG=(--timeout "${PYTEST_TIMEOUT}")
fi

TESTS_ARG=()
if [[ -n "${TESTS_LIST}" ]]; then
  # shellcheck disable=SC2206
  TESTS_ARG=(--tests ${TESTS_LIST})
fi

IGNORE_ARG=()
if [[ -n "${IGNORE_LIST}" ]]; then
  # shellcheck disable=SC2206
  for p in ${IGNORE_LIST}; do
    IGNORE_ARG+=(--ignore "$p")
  done
fi

echo "[$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID] Commit: $COMMIT"
echo "Repo: $REPO"
echo "Results: $OUTDIR"
echo "Host: $(hostname)"

python "$RUNNER" \
  --repo "$REPO" \
  --commit "$COMMIT" \
  --conda-env "$CONDA_ENV" \
  --results "$OUTDIR" \
  --run "$RUN_MODE" \
  --mutant-history "$MUTANT_HISTORY" \
  "${TIMEOUT_ARG[@]}" \
  "${TESTS_ARG[@]}" \
  "${IGNORE_ARG[@]}" \
  "${EXTRA_HEAD[@]}"

echo "Done commit $COMMIT."

