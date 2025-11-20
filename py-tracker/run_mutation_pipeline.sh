#!/usr/bin/env bash
set -euo pipefail

# Simple pipeline:
#  1) pytest in program mode (baseline)
#  2) cmutate on src-path
#  3) python s3.py to run mutants

# Required:
#   --base-repo   root directory of the project
#   --src-path    .py file located within the project directory
#   --tests       pytest target (file/dir/glob)
#   --save-path   directory where results will be stored
#
# Optional:
#   --njobs N
#   --jid J
#   --timeout-min SECS
#   --sandbox-mode {copy,symlink}
#   --stop-on-kill


BASE_REPO=""
SRC_PATH=""
TESTS=""
SAVE_PATH=""

NJOBS=1
JID=0
TIMEOUT_MIN=10
SANDBOX_MODE="copy"
STOP_ON_KILL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-repo)
      BASE_REPO="$2"; shift 2;;
    --src-path)
      SRC_PATH="$2"; shift 2;;
    --tests)
      TESTS="$2"; shift 2;;
    --save-path)
      SAVE_PATH="$2"; shift 2;;

    --njobs)
      NJOBS="$2"; shift 2;;
    --jid)
      JID="$2"; shift 2;;
    --timeout-min)
      TIMEOUT_MIN="$2"; shift 2;;
    --sandbox-mode)
      SANDBOX_MODE="$2"; shift 2;;
    --stop-on-kill)
      STOP_ON_KILL=1; shift 1;;

    *)
      echo "Unknown argument: $1" >&2
      exit 1;;
  esac
done

if [[ -z "$BASE_REPO" || -z "$SRC_PATH" || -z "$TESTS" || -z "$SAVE_PATH" ]]; then
  echo "Usage: $0 --base-repo DIR --src-path FILE.py --tests TESTPATH --save-path DIR [options...]" >&2
  exit 1
fi

# Resolve paths (absolute)
BASE_REPO_ABS=$(realpath "$BASE_REPO")
SAVE_PATH_ABS=$(realpath -m "$SAVE_PATH")
mkdir -p "$SAVE_PATH_ABS"

# src-path can be relative to base-repo or absolute
if [[ "$SRC_PATH" = /* ]]; then
  SRC_ABS=$(realpath "$SRC_PATH")
else
  SRC_ABS=$(realpath -m "$BASE_REPO_ABS/$SRC_PATH")
fi

# Program results and mutants JSON inside save-path
PROG_JSON="$SAVE_PATH_ABS/pres.json"
MUTANTS_JSON="$SAVE_PATH_ABS/mutants.json"
MUTANTS_RES_DIR="$SAVE_PATH_ABS/mutants_res"
mkdir -p "$MUTANTS_RES_DIR"

echo "Base repo      : $BASE_REPO_ABS"
echo "Source file    : $SRC_ABS"
echo "Tests          : $TESTS"
echo "Save path      : $SAVE_PATH_ABS"
echo "Program results: $PROG_JSON"
echo "Mutants JSON   : $MUTANTS_JSON"
echo "Mutant outputs : $MUTANTS_RES_DIR"
echo

###############################################################################
# 1) Baseline program run with pytest
###############################################################################
echo "=== Step 1: pytest in program mode ==="
(
  cd "$BASE_REPO_ABS"
  pytest $TESTS \
    --mode program \
    --program-results "$PROG_JSON" \
    --timeout-min "$TIMEOUT_MIN"
)
echo

###############################################################################
# 2) Generate mutants with cmutate
###############################################################################
echo "=== Step 2: cmutate on source ==="
cmutate -t "$SRC_ABS" -o "$MUTANTS_JSON"
echo

###############################################################################
# 3) Mutant execution via s3.py
###############################################################################
echo "=== Step 3: run mutants with s3.py ==="

S3_ARGS=(
  --base-repo "$BASE_REPO_ABS"
  --src-path "$SRC_ABS"
  --tests "$TESTS"
  --prog-res "$PROG_JSON"
  --mutants "$MUTANTS_JSON"
  --save-path "$MUTANTS_RES_DIR"
  --njobs "$NJOBS"
  --jid "$JID"
  --timeout-min "$TIMEOUT_MIN"
  --sandbox-mode "$SANDBOX_MODE"
)

if [[ "$STOP_ON_KILL" -eq 1 ]]; then
  S3_ARGS+=( --stop-on-kill )
fi

python3 mutant-sandbox-runner.py "${S3_ARGS[@]}"

echo "Pipeline finished."
