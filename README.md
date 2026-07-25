# Hugging Face Workspace

Developer-authored scripts and artifacts for fine-tuning models and running them,
with trained weights published to the **Hugging Face Hub**. No system files or
caches live here (those stay in `~/.cache/huggingface`); conventions for how the
workspace is organized are in [`CLAUDE.md`](CLAUDE.md).

## Layout

```
HuggingFace/                            ← workspace root
├── README.md                           – this file: workspace overview + shared tooling
├── CLAUDE.md                           – workspace conventions (for AI assistants)
├── run.sh · run.ps1                    – launcher entry (Linux/macOS · Windows): prerequisite
│                                         checks + uv bootstrap, then hand-off to run.py
├── pyproject.toml                      – launcher dependencies (huggingface_hub, kaggle),
│                                         installed by the launcher via `uv sync`
├── run.py                              – the runner: target prompt, script resolution, and
│                                         execution (HF Jobs in-process; externals delegated)
├── baseservice.py                      – BaseService interface every service runner extends
├── <namespace>/                        – a Hugging Face namespace (a user or org, e.g. smallTech)
│   └── <model>/                        – one folder per model, named after its Hub repo id
│       ├── data-preparation/           – (when used) HF Jobs data prep:
│       │                                 prepare-data.py + prepare-data.config.json
│       ├── trainingandevaluation/      – (when used) HF Jobs trainer:
│       │                                 train.py + train.config.json
│       ├── inferenceandtesting/
│       │   ├── inference.py            – run/test the trained model from the Hub
│       │   └── inference.config.json   – how to run it (local; hf jobs alternative)
│       ├── README.md                   – that model's card: base model, dataset, config
│       └── externals.json              – external services used by this model
└── external/                           – external services used by models
    └── <service_name>/                 – one folder per external service (e.g. kaggle)
        ├── service.py                  – that service's runner: ONE class extending
        │                                 BaseService, no main() (huggingface needs none —
        │                                 its class is built into run.py)
        └── <reference_canonical_name>/ – referenced from a model's externals.json
            ├── data-preparation/
            │   ├── prepare-data.kaggle.ipynb   – e.g. Kaggle CPU kernel staging the data
            │   └── prepare-data.config.json    – kernel options (GPU, internet, mounts)
            ├── trainingandevaluation/
            │   ├── train.kaggle.ipynb          – Kaggle (free T4) trainer notebook
            │   ├── train.config.json           – kernel options + option reference
            │   ├── smoketest.kaggle.ipynb      – few-minute environment check
            │   └── smoketest.config.json
            ├── inferenceandtesting/            – (when used)
            └── externals.json          – ONLY if this external delegates an
                                          operation to yet another external
```

**How the structure maps to the Hub.** The path `<namespace>/<model>` *is* the
model's Hugging Face Hub repo id: the folder
`smallTech/rtdetrv2-r50vd-sportsmot-players/` corresponds to the Hub model
`huggingface.co/smallTech/rtdetrv2-r50vd-sportsmot-players`.

**Model folders vs. external/.** A model folder holds the first-party,
Hub-centric pipeline, one stage folder per step (`data-preparation`,
`trainingandevaluation`, `inferenceandtesting`) — stages exist only when
actually used. Anything that runs on another service — e.g. Kaggle's free
GPUs — lives under `external/<service>/<reference>/` with the same stage
layout, and the model's `externals.json` records which externals it uses and
for what.

**`externals.json` semantics — delegated operations only.** An `externals.json`
declares *operations delegated to another service*, nothing else. Resources an
external merely consumes (mounted datasets, kernel outputs, Hub downloads) are
NOT externals — they belong in the runnable's `*.config.json`. An external gets
its own `externals.json` only if it in turn delegates an operation to yet
another external; if it delegates nothing (like the current Kaggle external),
the file must not exist. Orchestration tooling resolves references from these
files, so a file that exists implies real delegated work. Trained weights are
**not** stored anywhere in the workspace; they are pushed to the Hub.

**config.json files.** Every runnable ships next to a `<name>.config.json`
declaring how it runs: for Kaggle, the kernel metadata (accelerator
`machine_shape`, internet, `dataset_sources`, `kernel_sources`) consumed by
the kaggle service runner; for anything run as a Hugging Face Job, the
`hf jobs uv run` options (the full CLI option list is collated as a reference
inside the config, e.g. `inferenceandtesting/inference.config.json`).

**Service runners.** Every runner implements the `BaseService` interface
(root `baseservice.py`): `run(script, model, type, name) -> run_id` (Kaggle:
`<user>/<kernel-slug>`; Hugging Face: the job id), `get_status(run_id)`,
`list_runs()`, `list_datasets()`, and `login(token=None)` (token-based or
interactive). Every method except `login` is guarded — the base class checks
`is_logged_in()` before delegating, so implementations cannot be reached
logged-out. A shared `load_config` helper enforces the single-config-source
rule. Each external service ships exactly one
`external/<service>/service.py` defining a single BaseService subclass —
class only, no main(), no argparse — which `run.py` imports dynamically and
calls **in-process** (no subprocess). E.g. `KaggleService` pushes a kernel
with the config's metadata after verifying every mount, using the kaggle
Python library directly (the kaggle CLI is only a wrapper around it). The
default runner for first-party scripts in model folders is the built-in
`HuggingFaceService` inside `run.py`, which submits Hugging Face Jobs via the
huggingface_hub library (the hf CLI is likewise only a wrapper) — no
service.py exists for huggingface.

## Models

- [`smallTech/rtdetrv2-r50vd-sportsmot-players`](smallTech/rtdetrv2-r50vd-sportsmot-players/) —
  RT-DETRv2 basketball player detector, plus ByteTrack tracking inference.
  Trains on Kaggle: [`external/kaggle/rtdetrv2-r50vd-sportsmot-players/`](external/kaggle/rtdetrv2-r50vd-sportsmot-players/).

## Running anything — `run.sh` / `run.ps1` → `run.py`

The shell/PowerShell launchers perform only **step 1** — verify prerequisites
(git, python; each failure names the failed test command and what to install),
install **uv** globally if missing, and `uv sync` the workspace
`pyproject.toml` (which provides the `huggingface_hub` and `kaggle` libraries
— the hf/kaggle CLIs are *not* prerequisites, they are wrappers around these
libraries) — then delegate to `run.py` inside that environment, which hosts
the rest so both platforms behave identically: **2.** take the target
`<namespace>/<model>/<type>/<script_name>` from the argument or prompt for
it; **3.** resolve the script — the model folder's own `<type>/` first, then
every external referenced by the model's `externals.json`, erroring on zero
matches (no-script-found) or several (ambiguity); **4.** run it — the owning
service's BaseService class is loaded (dynamically imported for externals,
built-in HuggingFaceService for first-party scripts) and its `run(...)` is
called in-process.

```bash
./run.sh <namespace>/<model>/data-preparation/prepare-data      # CPU data staging (one-time)
./run.sh <namespace>/<model>/trainingandevaluation/smoketest    # few-minute environment check
./run.sh <namespace>/<model>/trainingandevaluation/train        # the training run
# Windows: .\run.ps1 <same argument>
```

For Kaggle targets, the service runner verifies every mounted source exists
(and that staging kernels are COMPLETE) before spending GPU time, then pushes
the kernel.

Recommended order for a first run: `prepare-data` (stages the training data
inside Kaggle — no GPU quota), then `smoketest` (validates GPU/torch, deps,
token, data mount in minutes), then the real training run.

One-time Kaggle auth setup (the library itself is installed by the launcher):

```bash
# Kaggle gives you EITHER a kaggle.json ({"username","key"}) OR a single
# access token (KGAT_...). Put whichever you have in ~/.kaggle/ :
#   ~/.kaggle/kaggle.json     (chmod 600)    or    ~/.kaggle/access_token  (chmod 600)
uv run kaggle config view     # should print your username + auth_method
```

Then create a **private** Kaggle dataset named `external-secrets` containing a
file called `secrets` with `HF_TOKEN=<write token>` (KEY=VALUE lines; add more
keys later as needed). This is a one-time step done on Kaggle directly.

**How the token reaches Kaggle.** Kaggle Secrets get dropped whenever the CLI
pushes a new notebook version, so instead the HF token lives in the private
`external-secrets` dataset. The launcher attaches that dataset in the kernel
metadata — dataset references *do* persist across versions — and notebooks scan
`/kaggle/input` for the `secrets` file (the mount layout has changed before;
currently `/kaggle/input/datasets/<owner>/<slug>/`). If the dataset can't be
read, training notebooks fall back to saving the model to the kernel **Output**
tab instead of pushing to the Hub.

**How training data reaches Kaggle.** Bulk-downloading many small files from
the Hub mid-session proved unreliable (CDN burst protection), so data is staged
**once, inside Kaggle**: a CPU-only `prepare-data` kernel mirrors the dataset
and outputs a single tar; training kernels mount that kernel's output
(`kernel_sources` in their config.json) and untar it locally at startup.

## Prerequisites

Only **git** and **python ≥ 3.10** need to be pre-installed — the launcher
installs uv and the Python libraries itself. What you must set up once is
*authentication*:

- **Hugging Face**: a login token with write access to the target namespace
  (`uv run hf auth login`, or an existing `~/.cache/huggingface/token`) — the
  trained model is pushed to the Hub.
- **Kaggle**: credentials in `~/.kaggle/` (see the auth setup above).

## Inference

Every model ships `inferenceandtesting/inference.py`, which loads the trained
model from the Hub and runs it. It auto-selects CUDA / Apple MPS / CPU, so it
runs locally on modest hardware — only *training* needs a dedicated GPU. See
each model's README.
