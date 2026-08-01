# Machine Learning Master — ML-operations workspace

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

Hub repos are tracked as **git submodules**, all under the `hf/` folder and
grouped by repo type: `hf/<namespace>/models/<model>/` is a submodule of the
HF model repo, `hf/<namespace>/spaces/<space>/` of the HF Space repo, and
`hf/<namespace>/datasets/<dataset>/` of the HF dataset repo (cloned with
`GIT_LFS_SKIP_SMUDGE=1` — weights stay as LFS pointers, never materialized
locally; gated repos need HF git credentials). Init only the submodules you
need — via the launcher's `load|unload <models|spaces|datasets>
<namespace> <name>` operations (`list namespaces` shows the account's namespaces and
which are loaded). Launcher targets keep the `<namespace>/<model>/...`
syntax; lib/main.py maps them to `hf/<namespace>/models/<model>/`. Stage subfolders — `data-preparation/`,
`training/`, `evaluation/`, `testing/`, … — exist only when actually exercised (no
runnables kept "for later"); each runnable sits next to a
`<name>.config.json`, the single source of its run options. A stage's main
runnable is named `index.*` — the folder itself names the operation — plus an
optional `smoketest.*` and, when the operation needs data staged for it, a
`prepare-data.*`. Staging is NOT dataset creation: `data-preparation/` exists
for models that actually produce a dataset; a model consuming an
already-prepared dataset (like the current one) has no such stage, and its
staging scripts live inside the operations they feed. Because the model
folder IS the Hub repo (submodule), its `README.md` is literally the model
card — card content only (frontmatter, benchmarks, usage); kernels update its
marker-delimited benchmark sections in place, so never hand-edit inside
markers, and always `git -C <submodule> pull` before editing (kernels and
HfApi uploads commit to the HF repo outside your local checkout). Repo-setup
and pipeline documentation belongs in the workspace README. `externals.json`
also lives in the model repo.

`externals.json` (model level) lists operations DELEGATED to other services:
entries of `{service, reference, path, used_for, notes}` pointing into the
service's root folder (e.g. `kaggle/`). `used_for` is a list of operation objects
`{"name": "<operation>", "artifacts": [{"name": ..., "type": "script" |
"result"}]}` — `script` artifacts are the external's runnables (the
declarations ARE the search space: an undeclared script on disk is invisible
to resolution),
`result` artifacts are operation outputs living in the model repo at
`.eval_results/<operation>.yaml` (the Hugging Face eval-results location). Resources merely consumed — datasets,
kernel outputs, Hub artifacts — belong in the runnable's config.json, never
here. Orchestration tooling resolves references from this file.

`<service>/<namespace>/jobs/<reference>/` (at the repo root, `<namespace>`
being the account's username on that service) mirrors the same stage
layout for anything that runs on another service (`kaggle/`, …). An external gets
its own externals.json only if it delegates further; the current Kaggle
external delegates nothing, so it has none.

All scripts must be self-explanatory via comments.

## Conventions

- Python runnables are self-contained uv scripts (PEP 723); first-party ones
  run as Hugging Face Jobs with the options in the adjacent config.json.
- Launcher: `./mlops.sh <operation> <entity> [args]` (Windows:
  `.\mlops.ps1`) — operations are verbs, the entity is their first
  mandatory argument: `list <namespaces|models|spaces|datasets|jobs>`;
  `load|unload <models|spaces|datasets> <namespace> <name>` (materialize /
  deinit the Hub repo's submodule under `hf/`); `git <models|spaces|datasets>
  <namespace> <name> <git args...>` (proxy a git command to that submodule);
  `execute jobs <namespace>/<model>/<type>/<script_name>`;
  `status jobs <run_id> [service]`; `create|delete
  <models|spaces|datasets|jobs> <service> <name>` (create/delete the entity
  on the remote service; delete is destructive and remote-only — local
  submodules are removed via `unload`); and `help [operation [entity]]`.
  The wrappers only
  check prerequisites (git, python), install uv, and `uv sync`; `lib/main.py`
  for `execute jobs` resolves the target (model folder first, then
  the model's externals.json) and runs it; the listing operations call the
  service methods (credentials are checked by each service's login(), not
  upfront). Details are in their comments.
- Every service runner is one class extending BaseService (workspace
  `lib/baseservice.py` — the docstring is the contract): exactly one service.py
  per service — `hf/service.py` for huggingface (Hugging Face Jobs),
  `<service>/service.py` for externals — class only (no main, no
  argparse, no import-time side effects), imported dynamically by lib/main.py and
  run in-process.
- Auth: lib/main.py `load_dotenv()`s the workspace `.env` (template
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
