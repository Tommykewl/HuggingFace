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
├── kaggle_train.sh                     – shared launcher: trains any model on Kaggle's free GPU
└── <namespace>/                        – a Hugging Face namespace (a user or org, e.g. smallTech)
    └── <model>/                        – one folder per model, named after its Hub repo id
        ├── trainer.py                  – Hugging Face Jobs trainer (uv script)
        ├── trainer.kaggle.ipynb        – Kaggle (free T4) trainer notebook
        ├── inference.py                – run/test the trained model from the Hub
        └── README.md                   – that model's card: base model, dataset, config
```

**How the structure maps to the Hub.** The path `<namespace>/<model>` *is* the
model's Hugging Face Hub repo id. For example the folder
`smallTech/rtdetrv2-r50vd-sportsmot-players/` corresponds to the Hub model
`huggingface.co/smallTech/rtdetrv2-r50vd-sportsmot-players`. The `<namespace>`
level groups all models owned by one user or organization; you can have several
model folders under it, and several namespaces in the workspace.

**What's shared vs. per-model.**

- *Workspace root* holds things common to every model: this README, the
  conventions in `CLAUDE.md`, and the generic `kaggle_train.sh` launcher (it takes
  a `<namespace>/<model>` argument, so one copy serves all models).
- *Each model folder* is self-contained and holds only that model's artifacts —
  the two trainers (one for Hugging Face Jobs, one for Kaggle), the inference
  script, and a model-specific README. Trained weights are **not** stored here;
  they are pushed to the Hub. No caches or credentials live in the workspace
  either (see [`CLAUDE.md`](CLAUDE.md)).

The four files every model folder contains, in the order you'd use them:

| File | Role |
|---|---|
| `trainer.py` | Fine-tunes the model on Hugging Face Jobs (`hf jobs uv run`), pushing the result to the Hub. |
| `trainer.kaggle.ipynb` | The same training run as a Kaggle notebook, tuned for the free T4 GPU. Launched via the root `kaggle_train.sh`. |
| `inference.py` | Loads the finished model from the Hub and runs it (locally on CPU/GPU/MPS). |
| `README.md` | Documents the model: base checkpoint, dataset, training configuration, and usage. |

## Models

- [`smallTech/rtdetrv2-r50vd-sportsmot-players`](smallTech/rtdetrv2-r50vd-sportsmot-players/) —
  RT-DETRv2 basketball player detector, plus ByteTrack tracking inference.

## Training a model

Each model can be trained two ways; both push the finished model to the Hub.

### Hugging Face Jobs (managed GPU, prepaid credits)

```bash
hf jobs uv run <namespace>/<model>/trainer.py --flavor a10g-large --timeout 6h \
  --detach --secrets "HF_TOKEN=$(hf auth token)"
```

### Kaggle free GPU — `kaggle_train.sh`

Runs a model's `trainer.kaggle.ipynb` on Kaggle's free T4, headless. It is generic
and takes the model path as its argument:

```bash
./kaggle_train.sh <namespace>/<model>      # e.g. smallTech/rtdetrv2-r50vd-sportsmot-players
```

If you omit the argument it prompts for it, and it verifies the folder and its
`trainer.kaggle.ipynb` exist before doing anything.

One-time Kaggle setup:

```bash
pip install --user kaggle
# Auth: Kaggle gives you EITHER a kaggle.json ({"username","key"}) OR a single
# access token (KGAT_...). Put whichever you have in ~/.kaggle/ :
#   ~/.kaggle/kaggle.json     (chmod 600)    or    ~/.kaggle/access_token  (chmod 600)
kaggle config view            # should print your username + auth_method
```

Then create a **private** Kaggle dataset named `external-secrets` containing a
file called `secrets` with `HF_TOKEN=<write token>` (KEY=VALUE lines; add more
keys later as needed). This is a one-time step done on Kaggle directly.

**How the token reaches Kaggle.** Kaggle Secrets get dropped whenever the CLI
pushes a new notebook version, so instead the HF token lives in the private
`external-secrets` dataset. The launcher references that dataset in the kernel
metadata — dataset references *do* persist across versions — and the notebook
reads `HF_TOKEN` from `/kaggle/input/external-secrets/secrets`.

`kaggle_train.sh` then does a single push: it verifies the `external-secrets`
dataset exists, uploads the notebook with that dataset attached, and queues the
GPU run, which trains and pushes the model to the Hub. Monitor with
`kaggle kernels status <user>/<model>`. (If the dataset can't be read, the
notebook falls back to saving the model to the kernel **Output** tab.)

## Prerequisites

- **`hf` CLI**, logged in (`hf auth whoami`) with write access to the target
  namespace — used by both training paths to push the model.
- **`kaggle` CLI** (for the Kaggle path), authenticated via `~/.kaggle/`.

## Inference

Every model ships an `inference.py` that loads the trained model from the Hub and
runs it. It auto-selects CUDA / Apple MPS / CPU, so it runs locally on modest
hardware — only *training* needs a dedicated GPU. See each model's README.
