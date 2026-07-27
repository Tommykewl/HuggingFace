# Hugging Face Workspace

## Non-negotiable rule

Get every plan (any series of scripts and operations) reviewed by the user
before executing it — runs here consume GPU quota and real money. Autonomy
covers only small mechanical steps inside an approved plan (create a folder,
run a single command, retry a flaky operation), never deciding or changing the
plan. If an approved approach proves infeasible, STOP and present the
alternatives with tradeoffs — do not build a replacement strategy unprompted.

The repo holds only developer-authored artifacts: no system files, caches, or
tokens (those stay in `~/.cache/huggingface`); no trained weights (those go to
the Hub). Temp data goes in the session scratchpad and is deleted once used.

## Layout

Model folder path = Hub repo id: `<namespace>/<model>/` (e.g.
`smallTech/rtdetr-sportsmot`). Stage subfolders — `data-preparation/`,
`training/`, `evaluation/`, `testing/`, … — exist only when actually exercised (no
runnables kept "for later"); each runnable sits next to a
`<name>.config.json`, the single source of its run options. A stage's main
runnable is named `index.*` — the folder itself names the operation — plus an
optional `smoketest.*` and, when the operation needs data staged for it, a
`prepare-data.*`. Staging is NOT dataset creation: `data-preparation/` exists
for models that actually produce a dataset; a model consuming an
already-prepared dataset (like the current one) has no such stage, and its
staging scripts live inside the operations they feed. Each model folder's
`README.md` IS the Hub model card, uploaded verbatim — card content only
(frontmatter, benchmarks, usage); kernels update its marker-delimited
benchmark sections in place, so never hand-edit inside markers. Repo-setup
and pipeline documentation belongs in the workspace README, not the model
README.

`externals.json` (model level) lists operations DELEGATED to other services:
entries of `{service, reference, path, used_for, notes}` pointing into
`external/`. Resources merely consumed — datasets, kernel outputs, Hub
artifacts — belong in the runnable's config.json, never here. Orchestration
tooling resolves references from this file.

`external/<service>/<reference>/` mirrors the same stage layout for anything
that runs on another service (`kaggle`, `local`, …). An external gets its own
externals.json only if it delegates further; the current Kaggle external
delegates nothing, so it has none.

All scripts must be self-explanatory via comments.

## Conventions

- Python runnables are self-contained uv scripts (PEP 723); first-party ones
  run as Hugging Face Jobs with the options in the adjacent config.json.
- Launcher: `./run.sh <namespace>/<model>/<type>/<script_name>` (Windows:
  `.\run.ps1`). The wrappers only check prerequisites (git, python), install
  uv, and `uv sync`; `run.py` resolves the target (model folder first, then
  the model's externals.json) and runs it. Details are in their comments.
- Every service runner is one class extending BaseService (root
  `baseservice.py` — the docstring is the contract): exactly one
  `external/<service>/service.py`, class only (no main, no argparse, no
  import-time side effects), imported dynamically by run.py and run
  in-process. Hugging Face Jobs is the built-in HuggingFaceService in run.py
  — no service.py for huggingface.
- Auth: run.py `load_dotenv()`s the workspace `.env` (template
  `.env.example`) at startup; every public service method calls the
  service's login() first, which mandates that service's credential env
  vars (HF_TOKEN; KAGGLE_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY) and exits
  naming what's missing. Env vars are the only login route — no
  interactive or stored-credential fallbacks.
- Call the service Python libraries (huggingface_hub, kaggle) directly; never
  subprocess the hf/kaggle CLIs or require them as prerequisites — they are
  wrappers around those libraries.
- Kaggle kernel naming: the runner derives the slug as `<type>-<name>`
  (e.g. `training-index`); a config `slug` field overrides —
  needed only when the derived name deviates from the kernel's actual name on
  Kaggle or exceeds Kaggle's 50-char slug limit (the API rejects longer with
  a bare 400, and the runner now guards this). Kernels are grouped in a
  Kaggle collection named after the model — maintained in the UI, the kaggle
  API has no collections endpoint. The runner verifies every dataset source
  exists and every kernel source is COMPLETE before pushing — never spend GPU
  time discovering a missing mount.
- The HF token reaches Kaggle via a private Kaggle dataset `external-secrets`
  (a `secrets` file of KEY=VALUE lines), listed in each kernel config's
  `dataset_sources`. The Kaggle runner's `_setup()` creates it from the
  HF_TOKEN env var when missing (extra keys are added on Kaggle directly);
  it injects nothing into kernels and warns when a config omits the dataset.
  Kaggle Secrets are unusable: dropped on every CLI push. Notebooks scan
  `/kaggle/input` for mounts instead of hard-coding paths (the layout has
  changed before).
- GPU kernel configs must pin `"machine_shape": "NvidiaTeslaT4"` — the API
  default GPU is a P100 (sm_60), for which Kaggle's torch ships no CUDA
  kernels ("no kernel image is available for execution on the device").
- Training data is staged once inside Kaggle (a CPU staging kernel
  mounted via `kernel_sources`, or a Kaggle Dataset), never bulk-downloaded
  from the Hub during training — that once consumed an entire GPU session,
  and residential IPs get CDN-throttled. The in-session download is a
  loud-warning fallback only.
- Run the matching Kaggle smoketest notebook (a few minutes) before any long
  training run: it validates GPU/torch kernels, deps, Hub auth, the
  staged-data mount, and a mock training pass.
