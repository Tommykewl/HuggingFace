#!/usr/bin/env bash
#
# run.sh — workspace launcher (Linux / macOS; Windows users: run.ps1).
#
# Runs one of a model's pipeline scripts by delegating to the service that
# hosts it:
#
#   ./run.sh <namespace>/<model>/<type>/<script_name>
#
#   <namespace>    Hugging Face namespace, e.g. smallTech
#   <model>        model folder = Hub repo id, e.g. rtdetrv2-r50vd-sportsmot-players
#   <type>         one of: data-preparation | trainingandevaluation | inferenceandtesting
#   <script_name>  runnable's base name, e.g. train, smoketest, prepare-data
#
# Examples:
#   ./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/data-preparation/prepare-data
#   ./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/trainingandevaluation/smoketest
#   ./run.sh smallTech/rtdetrv2-r50vd-sportsmot-players/trainingandevaluation/train
#
# This shell wrapper performs ONLY step 1 (prerequisite checks + environment
# setup) and then delegates to run.py, which hosts the remaining steps:
#   1. verify_prerequisites — git, python3 (here); then ensure_uv installs uv
#      if missing and `uv sync`s the workspace pyproject.toml, which provides
#      the huggingface_hub and kaggle libraries — the hf/kaggle CLIs are NOT
#      prerequisites, they are just wrappers around these libraries.
#   2. ask_target           — run.py: target from argv or prompt.
#   3. resolve_script       — run.py: model folder first, then externals.json;
#                             errors on none-found and on ambiguity.
#   4. run / delegate       — run.py: external services go to their runner
#                             (external/<service>/service.py); first-party
#                             scripts run as Hugging Face Jobs in-process via
#                             the huggingface_hub library (the hf CLI is just
#                             a wrapper around it, so no service.py for HF).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TYPES="data-preparation trainingandevaluation inferenceandtesting"

# ---------------------------------------------------------------------------
# Step 1 — verify prerequisites.
# Each requirement prints [OK]/[FAIL]; a failure shows the exact test command
# that failed and what to install, and the launcher exits before doing work.
# ---------------------------------------------------------------------------
_PREREQ_FAILED=0

_require() {
  # _require <label> <install hint> <test command>
  local label="$1" hint="$2" test_cmd="$3"
  if eval "$test_cmd" >/dev/null 2>&1; then
    echo "  [OK]   $label"
  else
    echo "  [FAIL] $label"
    echo "         test command failed: $test_cmd"
    echo "         install: $hint"
    _PREREQ_FAILED=1
  fi
}

verify_prerequisites() {
  # pip --user / pipx install their console scripts into bin dirs that are
  # often not on PATH; add the common ones so the checks (and the service
  # runners we delegate to) can find hf/kaggle.
  export PATH="$HOME/.local/bin:$HOME/Library/Python/3.12/bin:$PATH"

  echo "Checking prerequisites..."
  _require "git" \
    "https://git-scm.com (macOS: xcode-select --install; Debian/Ubuntu: apt install git)" \
    "git --version"
  _require "python3 >= 3.10" \
    "https://www.python.org or your OS package manager" \
    "python3 -c 'import sys; assert sys.version_info >= (3, 10)'"
  if [ "$_PREREQ_FAILED" -ne 0 ]; then
    echo "Prerequisites NOT met — install the items marked [FAIL] above and re-run."
    exit 1
  fi
  echo "All requirements met."
}

# ---------------------------------------------------------------------------
# Step 1b — environment setup with uv.
# Install uv globally if it isn't already, then install the workspace
# dependencies (pyproject.toml: huggingface_hub, kaggle) into the project
# environment with `uv sync`. run.py and the service runners then execute
# inside that environment, importing the libraries directly.
# ---------------------------------------------------------------------------
ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it globally (https://astral.sh/uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "uv installation failed — install it manually from https://docs.astral.sh/uv/"; exit 1; }
  fi
  echo "Installing workspace dependencies (uv sync)..."
  uv sync --project "$ROOT" --quiet
}

# ---------------------------------------------------------------------------
# Steps 2-4 live in run.py (shared with run.ps1 so both platforms behave
# identically), executed inside the uv-managed environment.
# ---------------------------------------------------------------------------
main() {
  verify_prerequisites
  ensure_uv
  exec uv run --project "$ROOT" python "$ROOT/run.py" "${1:-}"
}

main "${1:-}"
