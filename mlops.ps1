# mlops.ps1 — workspace launcher (Windows; Linux/macOS users: mlops.sh).
#
# Runs one workspace operation by delegating to lib/main.py:
#
#   .\mlops.ps1 <operation> <entity> [args]
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
#   .\mlops.ps1 execute jobs smallTech/rtdetr-sportsmot/training/smoketest
#   .\mlops.ps1 execute jobs smallTech/rtdetr-sportsmot/training/index
#   .\mlops.ps1 list jobs
#   .\mlops.ps1 list namespaces
#
# This PowerShell wrapper performs ONLY step 1 (prerequisite checks +
# environment setup) and then delegates to lib/main.py, which hosts the remaining
# steps (shared with mlops.sh so both platforms behave identically):
#   1. Test-Prerequisites — git, python (here); then Install-Uv installs uv
#      if missing and `uv sync`s the workspace pyproject.toml, which provides
#      the huggingface_hub and kaggle libraries — the hf/kaggle CLIs are NOT
#      prerequisites, they are just wrappers around these libraries.
#   2. parse_operation    — lib/main.py: operation from argv (none -> help).
#   3. dispatch           — lib/main.py: `execute jobs` resolves the script (model folder
#                           first, then externals.json; errors on none-found
#                           and on ambiguity) and hands it to its service —
#                           external services via their runner
#                           (<service>/service.py); first-party
#                           scripts as Hugging Face Jobs in-process via the
#                           huggingface_hub library (the hf CLI is just a
#                           wrapper around it, so no service.py for HF).
#                           Listing operations call the service methods.

# Operations are multi-word (e.g. "run list") — collect every argument and
# forward them all to lib/main.py.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Operation = @())

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
# Quiet when everything is in place; a failure prints [FAIL] with the exact
# test command and what to install, and the launcher exits before doing work.
# ---------------------------------------------------------------------------
function Test-Prerequisites {
    $failed = $false
    $python = Get-Python

    function Test-Requirement([string]$Label, [string]$Hint, [scriptblock]$Check, [string]$CheckText) {
        # Quiet on success — output only when a requirement is missing.
        $ok = $false
        try { $ok = (& $Check) } catch { $ok = $false }
        if (-not $ok) {
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
}

# ---------------------------------------------------------------------------
# Step 1b — environment setup with uv.
# Install uv globally if it isn't already, then install the workspace
# dependencies (pyproject.toml: huggingface_hub, kaggle) into the project
# environment with `uv sync`. lib/main.py and the service runners then execute
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
    # Quiet on success; uv prints its own errors on failure.
    uv sync --project $Root --quiet
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---------------------------------------------------------------------------
# Steps 2-4 live in lib/main.py (shared with mlops.sh so both platforms behave
# identically), executed inside the uv-managed environment.
# ---------------------------------------------------------------------------
Test-Prerequisites
Install-Uv
uv run --project $Root python (Join-Path $Root "lib" "main.py") @Operation
exit $LASTEXITCODE
