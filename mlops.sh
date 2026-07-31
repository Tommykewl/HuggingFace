#!/usr/bin/env bash
#
# mlops.sh — workspace launcher (Linux / macOS; Windows users: mlops.ps1).
#
# Runs one workspace operation by delegating to lib/main.py:
#
#   ./mlops.sh <operation> <entity> [args]
#
#   list <namespaces|models|spaces|datasets|jobs>  the account's <entity>, per service (Hub repos marked loaded/not-loaded)
#   load <models|spaces|datasets> <namespace> <name>    materialize the Hub repo as a submodule under hf/
#   unload <models|spaces|datasets> <namespace> <name>  deinit the submodule (empties it locally, stays registered)
#   git <models|spaces|datasets> <namespace> <name> <git args...>  proxy a git command to that submodule
#   execute jobs <namespace>/<model>/<type>/<script_name>  submit a runnable
#   status jobs <run_id> [service]                 one run's status (default: huggingface)
#   help [operation [entity]]                      usage overview, or one operation's details
#
#   <namespace>    Hugging Face namespace, e.g. smallTech
#   <model>        model folder = Hub repo id, e.g. rtdetr-sportsmot
#   <type>         one of: data-preparation | training | evaluation | testing
#   <script_name>  runnable's base name, e.g. train, smoketest, prepare-data
#
# Examples:
#   ./mlops.sh execute jobs smallTech/rtdetr-sportsmot/training/smoketest
#   ./mlops.sh execute jobs smallTech/rtdetr-sportsmot/training/index
#   ./mlops.sh list jobs
#   ./mlops.sh list namespaces
#
# This shell wrapper performs ONLY step 1 (prerequisite checks + environment
# setup) and then delegates to lib/main.py, which hosts the remaining steps:
#   1. verify_prerequisites — git, python3 (here); then ensure_uv installs uv
#      if missing and `uv sync`s the workspace pyproject.toml, which provides
#      the huggingface_hub and kaggle libraries — the hf/kaggle CLIs are NOT
#      prerequisites, they are just wrappers around these libraries.
#   2. parse_operation      — lib/main.py: operation from argv (none -> help).
#   3. dispatch             — lib/main.py: `execute jobs` resolves the script (model folder
#                             first, then externals.json; errors on none-found
#                             and on ambiguity) and hands it to its service —
#                             external services via their runner
#                             (<service>/service.py); first-party
#                             scripts as Hugging Face Jobs in-process via the
#                             huggingface_hub library (the hf CLI is just a
#                             wrapper around it, so no service.py for HF).
#                             Listing operations call the service methods.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TYPES="data-preparation training evaluation testing"

# ---------------------------------------------------------------------------
# Step 1 — verify prerequisites.
# Quiet when everything is in place; a failure prints [FAIL] with the exact
# test command and what to install, and the launcher exits before doing work.
# ---------------------------------------------------------------------------
_PREREQ_FAILED=0

_require() {
  # _require <label> <install hint> <test command>
  # Quiet on success — output only when a requirement is missing.
  local label="$1" hint="$2" test_cmd="$3"
  if ! eval "$test_cmd" >/dev/null 2>&1; then
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
}

# ---------------------------------------------------------------------------
# Step 1b — environment setup with uv.
# Install uv globally if it isn't already, then install the workspace
# dependencies (pyproject.toml: huggingface_hub, kaggle) into the project
# environment with `uv sync`. lib/main.py and the service runners then execute
# inside that environment, importing the libraries directly.
# ---------------------------------------------------------------------------
ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found — installing it globally (https://astral.sh/uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "uv installation failed — install it manually from https://docs.astral.sh/uv/"; exit 1; }
  fi
  # Quiet on success; uv prints its own errors on failure (set -e exits).
  uv sync --project "$ROOT" --quiet
}

# ---------------------------------------------------------------------------
# Steps 2-4 live in lib/main.py (shared with mlops.ps1 so both platforms behave
# identically), executed inside the uv-managed environment.
# ---------------------------------------------------------------------------
main() {
  verify_prerequisites
  ensure_uv
  # Forward ALL arguments — operations are multi-word (e.g. "run list").
  exec uv run --project "$ROOT" python "$ROOT/lib/main.py" "$@"
}

main "$@"
