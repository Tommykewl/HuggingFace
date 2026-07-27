# run.ps1 — workspace launcher (Windows; Linux/macOS users: run.sh).
#
# Runs one of a model's pipeline scripts by delegating to the service that
# hosts it:
#
#   .\run.ps1 <namespace>/<model>/<type>/<script_name>
#
#   <namespace>    Hugging Face namespace, e.g. smallTech
#   <model>        model folder = Hub repo id, e.g. rtdetr-sportsmot
#   <type>         one of: data-preparation | training | evaluation | testing
#   <script_name>  runnable's base name, e.g. train, smoketest, prepare-data
#
# Examples:
#   .\run.ps1 smallTech/rtdetr-sportsmot/training/prepare-data
#   .\run.ps1 smallTech/rtdetr-sportsmot/training/smoketest
#   .\run.ps1 smallTech/rtdetr-sportsmot/training/index
#
# This PowerShell wrapper performs ONLY step 1 (prerequisite checks +
# environment setup) and then delegates to run.py, which hosts the remaining
# steps (shared with run.sh so both platforms behave identically):
#   1. Test-Prerequisites — git, python (here); then Install-Uv installs uv
#      if missing and `uv sync`s the workspace pyproject.toml, which provides
#      the huggingface_hub and kaggle libraries — the hf/kaggle CLIs are NOT
#      prerequisites, they are just wrappers around these libraries.
#   2. ask_target         — run.py: target from argv or prompt.
#   3. resolve_script     — run.py: model folder first, then externals.json;
#                           errors on none-found and on ambiguity.
#   4. run / delegate     — run.py: external services go to their runner
#                           (external/<service>/service.py); first-party
#                           scripts run as Hugging Face Jobs in-process via
#                           the huggingface_hub library (the hf CLI is just
#                           a wrapper around it, so no service.py for HF).

param([string]$Target = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Types = @("data-preparation", "training", "evaluation", "testing")

# Python launcher name differs across platforms; resolve it once.
function Get-Python {
    foreach ($candidate in @("python", "python3", "py")) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) { return $candidate }
    }
    return $null
}

# ---------------------------------------------------------------------------
# Step 1 — verify prerequisites.
# Each requirement prints [OK]/[FAIL]; a failure shows the exact test command
# that failed and what to install, and the launcher exits before doing work.
# ---------------------------------------------------------------------------
function Test-Prerequisites {
    Write-Host "Checking prerequisites..."
    $failed = $false
    $python = Get-Python

    function Test-Requirement([string]$Label, [string]$Hint, [scriptblock]$Check, [string]$CheckText) {
        $ok = $false
        try { $ok = (& $Check) } catch { $ok = $false }
        if ($ok) {
            Write-Host "  [OK]   $Label"
        } else {
            Write-Host "  [FAIL] $Label"
            Write-Host "         test command failed: $CheckText"
            Write-Host "         install: $Hint"
            Set-Variable -Name failed -Value $true -Scope 1
        }
    }

    Test-Requirement "git" "https://git-scm.com/download/win" `
        { git --version *> $null; $LASTEXITCODE -eq 0 } "git --version"
    Test-Requirement "python >= 3.10" "https://www.python.org/downloads/windows/" `
        { if (-not $python) { return $false }
          & $python -c "import sys; assert sys.version_info >= (3, 10)" *> $null
          $LASTEXITCODE -eq 0 } "python -c `"import sys; assert sys.version_info >= (3, 10)`""
    if ($failed) {
        Write-Host "Prerequisites NOT met - install the items marked [FAIL] above and re-run."
        exit 1
    }
    Write-Host "All requirements met."
}

# ---------------------------------------------------------------------------
# Step 1b — environment setup with uv.
# Install uv globally if it isn't already, then install the workspace
# dependencies (pyproject.toml: huggingface_hub, kaggle) into the project
# environment with `uv sync`. run.py and the service runners then execute
# inside that environment, importing the libraries directly.
# ---------------------------------------------------------------------------
function Install-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv not found - installing it globally (https://astral.sh/uv)..."
        powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Host "uv installation failed - install it manually from https://docs.astral.sh/uv/"
            exit 1
        }
    }
    Write-Host "Installing workspace dependencies (uv sync)..."
    uv sync --project $Root --quiet
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---------------------------------------------------------------------------
# Steps 2-4 live in run.py (shared with run.sh so both platforms behave
# identically), executed inside the uv-managed environment.
# ---------------------------------------------------------------------------
Test-Prerequisites
Install-Uv
uv run --project $Root python (Join-Path $Root "run.py") $Target
exit $LASTEXITCODE
