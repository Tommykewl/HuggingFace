# Hugging Face Workspace

Holds only developer-authored artifacts. No system files (`HF_HOME`, caches,
tokens) — those stay in `~/.cache/huggingface`. Trained models go to the Hub, not
here. Temp data (dataset snapshots, sample frames, metadata) goes in the session
scratchpad and is deleted once used.

## Model folders

Path = Hub repo id: `<namespace>/<model>/` (e.g. `smallTech/rtdetrv2-...`). Each
contains exactly:

- `trainer.py` — Hugging Face Jobs trainer.
- `trainer.kaggle.ipynb` — same run, adapted for Kaggle free-tier.
- `inference.py` — test the trained model from the Hub.
- `README.md` — base model, dataset, config, usage.

All scripts must be self-explanatory via comments.

## Conventions

- Trainers are self-contained uv scripts (PEP 723), run with `hf jobs uv run <namespace>/<model>/trainer.py`.
- Kaggle: `./kaggle_train.sh <namespace>/<model>` (workspace root).
- Kaggle HF token comes from a private Kaggle dataset `external-secrets` (a
  `secrets` file of `KEY=VALUE` lines), referenced in kernel metadata and read at
  `/kaggle/input/external-secrets/secrets`. Maintained on Kaggle directly; not
  part of training. (Kaggle Secrets are unusable — dropped on each CLI push.)
