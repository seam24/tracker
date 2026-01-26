#!/bin/bash
#SBATCH --job-name=latentmu_commit
#SBATCH --output=/dev/null
#SBATCH --error=/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/logs/commit_%A_%a.err
#SBATCH --array=1-167
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

SRC_DIR="/mnt/aiongpfs/users/trostami/LatentMu/src"
REPO="/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/thefuck"
RUNNER="${SRC_DIR}/commit_test_runner.py"

# ONE shared output directory for all commits
RESULTS_BASE="/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/data/program_results"

MUTANT_HISTORY="/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/data/mutant_history.json"
COMMITS_FILE="/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/data/commits_to_run.txt"
CONDA_ENV="thefuck-32b-env"

PYTEST_TIMEOUT="20"
TESTS_LIST=""
IGNORE_LIST=""
RUN_MODE="both"

mkdir -p "$RESULTS_BASE"
mkdir -p "/mnt/aiongpfs/users/trostami/LatentMu/commits/thefuck-4b/logs"

source ~/miniconda3/etc/profile.d/conda.sh
cd "$SRC_DIR"

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

# IMPORTANT: shared output dir
OUTDIR="$RESULTS_BASE"

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
echo "Results dir (shared): $OUTDIR"
echo "Run mode: $RUN_MODE"

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
