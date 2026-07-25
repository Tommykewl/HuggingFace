# Hugging Face Workspace

## Non-negotiable rule

Before executing any plan (a series of scripts and operations), get the plan
reviewed by the user. This is a data-pipeline and model-training project: script
runs consume GPU quota and spawn long-running processes — true financial costs,
unlike ordinary application development. Autonomy covers only small mechanical
steps inside a user-approved plan (create a folder, run a single command, retry
a flaky operation); it never covers deciding or changing the plan itself. If an
approved approach turns out to be infeasible, STOP and present the alternatives
with their tradeoffs — do not build a replacement strategy unprompted.

Holds only developer-authored artifacts. No system files (`HF_HOME`, caches,
tokens) — those stay in `~/.cache/huggingface`. Trained models go to the Hub, not
here. Temp data (dataset snapshots, sample frames, metadata) goes in the session
scratchpad and is deleted once used.

## Model folders

Path = Hub repo id: `<namespace>/<model>/` (e.g. `smallTech/rtdetrv2-...`).
Stages are subfolders that exist only when actually used; each runnable sits
next to a `<name>.config.json` that declares how it runs:

- `data-preparation/` — (when used) `prepare-data.py` + `prepare-data.config.json`
  (Hugging Face Job for data prep).
- `trainingandevaluation/` — (when used) `train.py` (HF Jobs trainer, uv
  script) + `train.config.json` (hf jobs options: flavor, timeout, secrets, …).
- `inferenceandtesting/` — `inference.py` + `inference.config.json` (carries
  the full `hf jobs uv run` option list collated as a reference).
- `README.md` — base model, dataset, config, usage.
- `externals.json` — operations this model DELEGATES to external services:
  entries of `{service, reference, path, used_for, notes}` pointing into
  `external/`. Delegated operations only — resources merely consumed (datasets,
  kernel outputs, Hub artifacts) belong in the runnable's config.json, never
  here. Orchestration tooling resolves references from this file.

Do not keep unused runnables around "for later" — a stage folder appears when
its script is actually exercised (e.g. the current model trains only on Kaggle,
so it has no HF-Jobs train.py).

## external/ folders

`external/<service>/<reference_canonical_name>/` holds everything that runs on
another service (`kaggle`, `local`, …), mirroring the same stage layout:

- `data-preparation/`, `trainingandevaluation/`, `inferenceandtesting/` — each
  runnable + its `<name>.config.json`. For Kaggle notebooks the config declares
  kernel metadata (`enable_gpu`, `machine_shape`, `enable_internet`,
  `dataset_sources`, `kernel_sources`, optional pinned `slug`) which the
  kaggle service runner (`external/kaggle/service.py`) turns into
  kernel-metadata.json (bare slugs get the Kaggle username prefixed).
- `externals.json` — ONLY if this external delegates an operation to yet
  another external. Same delegated-operations-only semantics as the model-level
  file: consuming a resource (a mounted dataset, a kernel output, the Hub) is
  NOT delegation, so the current Kaggle external — which delegates nothing —
  has no externals.json at all.

Current example: `external/kaggle/rtdetrv2-r50vd-sportsmot-players/` (T4
training + smoketest + CPU data staging).

All scripts must be self-explanatory via comments.

## Conventions

- Python runnables are self-contained uv scripts (PEP 723); HF Jobs ones run
  with `hf jobs uv run <path>` using the options in the adjacent config.json.
- Launcher: `./run.sh <namespace>/<model>/<type>/<script_name>` (Linux/macOS;
  Windows: `.\run.ps1`). The shell wrappers do ONLY step 1 — prerequisite
  checks (git, python), installing uv globally if missing, and `uv sync`ing
  the workspace pyproject.toml (huggingface_hub + kaggle libraries) — then
  delegate to `run.py` inside that environment, which hosts step 2 (target
  from argv or prompt), step 3 (resolve: model folder first, then externals
  from externals.json — error on none-found or ambiguity) and step 4 (run).
  The hf/kaggle CLIs are never prerequisites: they are wrappers around the
  libraries pyproject.toml provides.
- Every service runner is a class implementing the BaseService interface
  (root `baseservice.py`): `run(...) -> run_id` (Kaggle `<user>/<kernel-slug>`,
  HF job id), `get_status(run_id)`, `list_runs()`, `list_datasets()`,
  `login(token=None)` (token or interactive), `is_logged_in()`. The base class
  guards every method except login with a login check (template methods:
  public concrete wrappers delegate to protected `_impl`s), plus a shared
  `load_config`. External services ship exactly one
  `external/<service>/service.py` defining a single BaseService subclass —
  class only: no main(), no argparse, no import-time side effects. run.py
  imports it dynamically and calls run() in-process (never a subprocess).
- Runners call service Python libraries directly — the hf and kaggle CLIs are
  wrappers around huggingface_hub and kaggle, so no subprocessing CLIs. The
  Hugging Face Jobs runner (default for first-party model-folder scripts) is
  the built-in HuggingFaceService in `run.py` (HfApi.run_uv_job with the
  config.json options) — no service.py exists for huggingface.
- Kaggle runner (KaggleService in `external/kaggle/service.py`): uses the
  kaggle library (KaggleApi) to build kernel metadata from the notebook's
  config.json. Kernel slugs: `train` keeps the model name;
  others get `<model>-<name>`; a config `slug` field overrides (used to pin
  the existing staging kernel). It verifies every dataset source exists and
  every kernel source is COMPLETE before pushing — never spend GPU time to
  discover a missing mount.
- Kaggle HF token comes from a private Kaggle dataset `external-secrets` (a
  `secrets` file of `KEY=VALUE` lines), listed in every kernel config's
  `dataset_sources` — the runner injects nothing and warns if it is missing
  from a config. Datasets
  mount at `/kaggle/input/datasets/<owner>/<slug>/`, kernel outputs at
  `/kaggle/input/notebooks/<owner>/<slug>/` (layout has changed before, so
  notebooks scan `/kaggle/input` instead of hard-coding paths). Maintained on
  Kaggle directly. (Kaggle Secrets are unusable — dropped on each CLI push.)
- Kaggle kernel configs must pin `"machine_shape": "NvidiaTeslaT4"` for GPU
  runs — the API default GPU is a P100 (sm_60), for which Kaggle's torch ships
  no CUDA kernels (fails with "no kernel image is available for execution on
  the device").
- Training data for Kaggle runs is staged once inside Kaggle (the kaggle
  external's CPU `prepare-data` kernel, mounted via `kernel_sources`; or a
  Kaggle Dataset), never bulk-downloaded from the Hub during training — a
  many-small-files Hub download once consumed an entire GPU session, and
  residential IPs get CDN-throttled. Notebooks auto-detect staged data and
  treat the in-session download strictly as a warned fallback.
- Run the Kaggle `smoketest` notebook (few minutes) before any multi-hour
  training run: it validates GPU/torch kernels, deps, Hub auth, the staged-data
  mount, and a mock training pass on the real image.
