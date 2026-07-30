# Hugging Face Workspace

Developer-authored scripts for fine-tuning models and running them. Trained
weights are published to the **Hugging Face Hub**, never stored here; no
system files or caches live here either (those stay in `~/.cache/huggingface`).
Workspace conventions are in [`CLAUDE.md`](CLAUDE.md).

## Layout

```
HuggingFace/                            ← workspace root
├── run.sh · run.ps1                    – launcher (Linux/macOS · Windows): prerequisite
│                                         checks + uv bootstrap, then hand-off to run.py
├── run.py                              – the runner: target prompt, script resolution, execution
├── baseservice.py                      – BaseService interface every service runner extends
├── pyproject.toml                      – launcher dependencies (huggingface_hub, kaggle)
├── <namespace>/                        – a Hub namespace (user or org, e.g. smallTech)
│   ├── models/<model>/                 – git SUBMODULE of the HF model repo (card, configs,
│   │                                     externals.json, .eval_results/ — weights stay LFS pointers)
│   └── spaces/<space>/                 – git SUBMODULE of the HF Space repo (demo apps)
│       — inside a model submodule:
│       ├── data-preparation/           – (when used) dataset-CREATION runnables — absent when
│       │                                 the dataset is already prepared, as for the current model
│       ├── training/                   – (when used) index.py (the trainer) + smoketest.py + configs
│       ├── evaluation/                 – (when used) index.py (the benchmark) + smoketest.py + configs
│       ├── testing/                    – (when used) performance-only benchmark (no ground truth)
│       ├── README.md                   – the model's card: base model, dataset, config, usage
│       └── externals.json              – operations this model delegates to external services
└── external/                           – everything that runs on another service
    └── <service>/                      – e.g. kaggle
        ├── service.py                  – the service's runner: one BaseService subclass
        └── <reference>/                – referenced from a model's externals.json;
            └── ...                       same stage layout, e.g. train.kaggle.ipynb + config
```

`<namespace>/models/<model>` is a **git submodule of the model's HF repo**:
`smallTech/models/rtdetr-sportsmot/` ↔
`huggingface.co/smallTech/rtdetr-sportsmot` (same for Spaces under
`<namespace>/spaces/<space>`). Submodules mean you init only what you work
on instead of pulling everything. A model folder
holds the first-party pipeline (run as Hugging Face Jobs); anything that runs
elsewhere — e.g. Kaggle's free GPUs — lives under
`external/<service>/<reference>/` with the same stage layout, referenced from
the model's `externals.json`. Every runnable has an adjacent
`<name>.config.json` declaring exactly how it runs.

## Models

- [`smallTech/rtdetr-sportsmot`](smallTech/models/rtdetr-sportsmot/) —
  RT-DETRv2 sports player detector (detection stage of a ByteTrack
  tracking-by-detection pipeline). The model folder holds the Hub model card
  (`README.md`, uploaded verbatim), `externals.json`, and the first-party
  `inference/` stage (the repo's Colab/Kaggle `notebook.ipynb` template).
  All other operations run on Kaggle:
  [`external/kaggle/rtdetr-sportsmot/`](external/kaggle/rtdetr-sportsmot/).

  | Stage (external/kaggle) | Runnables |
  |---|---|
  | `training/` | `prepare-data` (stages the train split in-network, CPU) · `smoketest` · `index` (the T4 trainer — pushes model + card to the Hub with reload verification) |
  | `evaluation/` | `prepare-data` (stages the unseen val split) · `smoketest` · `index` (accuracy benchmark: mAP/mAR + latency; publishes its card section) |
  | `testing/` | `prepare-data-1`/`-2` (full test split as two disjoint halves — 20 GB output cap) · `smoketest` · `index` (batched-fp16 performance benchmark; publishes its card section) |

  Run any stage end-to-end (staging first, then smoketest, then index):

  ```bash
  ./run.sh smallTech/rtdetr-sportsmot/training/prepare-data     # then training/smoketest, training/index
  ./run.sh smallTech/rtdetr-sportsmot/evaluation/prepare-data   # then evaluation/smoketest, evaluation/index
  ./run.sh smallTech/rtdetr-sportsmot/testing/prepare-data-1    # + prepare-data-2, then testing/smoketest, testing/index
  ```

## Setup

Only **git** and **python ≥ 3.10** need to be pre-installed — the launcher
installs uv and all Python libraries itself. After cloning, attach your two
accounts:

**1. Bootstrap the environment**

```bash
git clone https://github.com/Tommykewl/HuggingFace.git && cd HuggingFace
# init only the submodules you need (models/spaces are HF repos; weights stay
# as LFS pointers thanks to GIT_LFS_SKIP_SMUDGE; gated repos need HF git creds)
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init smallTech/models/rtdetr-sportsmot
./run.sh        # Windows: .\run.ps1 — installs uv, syncs deps; Ctrl-C at the target prompt
```

**2. Attach your accounts** — copy the template and fill in your tokens; the
launcher logs in with them automatically on first run:

```bash
cp .env.example .env    # gitignored — never committed
```

- `HF_TOKEN`: a token with **write** access to your target namespace, from
  <https://huggingface.co/settings/tokens>.
- `KAGGLE_TOKEN`: an access token from <https://www.kaggle.com/settings>
  (or uncomment `KAGGLE_USERNAME`/`KAGGLE_KEY` if you use a `kaggle.json`
  credential pair instead). The Kaggle account must be **phone-verified** —
  required for kernel internet + GPU.

`.env` is the only login route: if a required variable is missing, the
launcher exits naming it instead of prompting.

**3. Fine-tune** — for each model, in order:

```bash
./run.sh <namespace>/<model>/training/prepare-data        # once: stage training data inside Kaggle (CPU)
./run.sh <namespace>/<model>/training/smoketest        # minutes: validates GPU, deps, auth, mount
./run.sh <namespace>/<model>/training/index            # the real training run
```

## How it runs

The shell wrappers only verify prerequisites and set up the uv environment,
then delegate to `run.py`, which takes the target
`<namespace>/<model>/<type>/<script_name>` (argument or prompt), resolves the
script — the model folder first, then every external referenced by its
`externals.json` (only scripts declared as artifacts of that operation in
`used_for` are eligible), erroring on zero or multiple matches — and calls the owning
service's `run()` in-process. Services are classes implementing the
`BaseService` interface ([baseservice.py](baseservice.py)): the built-in
`HuggingFaceService` in `run.py` submits first-party scripts as Hugging Face
Jobs via the huggingface_hub library; externals ship one
`external/<service>/service.py` (e.g. [`KaggleService`](external/kaggle/service.py)).
The hf/kaggle CLIs are never used — they are just wrappers around these
libraries.

Before spending GPU time, the Kaggle runner verifies every mount declared in
the config exists (datasets present, staging kernels COMPLETE) and only then
pushes the kernel.

**How the HF token reaches Kaggle.** Kaggle Secrets are dropped whenever the
API pushes a new notebook version, so the token lives in a private
`external-secrets` Kaggle dataset instead — dataset references persist across
versions. The Kaggle runner creates this dataset automatically on first run
(a `secrets` file of `KEY=VALUE` lines, from the `HF_TOKEN` in your `.env`);
add further keys on Kaggle directly as needed. Notebooks scan `/kaggle/input`
for the `secrets` file (the mount layout has changed before). If the dataset
can't be read, training falls back to saving the model to the kernel
**Output** tab instead of pushing to the Hub.

**How training data reaches Kaggle.** Bulk-downloading many small files from
the Hub mid-session proved unreliable (CDN burst protection), so data is
staged **once, inside Kaggle**: a CPU-only staging kernel mirrors the
dataset and outputs a single tar; training kernels mount that output
(`kernel_sources` in their config) and untar it at startup.

## Benchmarking

Trained models are benchmarked on data they have never seen. For the current
model this too runs on Kaggle (the model folder delegates every operation),
under its `evaluation` stage: the `evaluation/index` T4 kernel pulls the
trained model from the Hub and scores it against the staged, annotated val
split — overall and per-sequence COCO mAP/mAR plus latency percentiles —
logging everything, writing `benchmarks.json` to the kernel output, and
publishing the results to the Hub model card (marker-delimited sections,
updated in place). A `testing` stage does the same on the test split —
performance metrics only, since SportsMOT withholds test ground truth. See
each model's README for the exact targets.
