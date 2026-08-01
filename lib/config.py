"""Workspace-wide constants shared by every ops module and lib/main.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the workspace root
HF_DIR = ROOT / "hf"                   # all Hugging Face git submodules live here
TYPES = ("data-preparation", "training", "evaluation", "testing")
# The operation catalog itself lives in two synchronized tables: SPECS in
# lib/main.py (grammar + dispatch); helptexts live in the operation classes.

# The three Hub repo kinds hf/ can hold, keyed by their entity word —
# which doubles as the hf/<ns>/ subfolder and the list(entity) string:
# (Hub repo_type for the API, Hub URL prefix).
REPO_KINDS = {
    "models": ("model", "https://huggingface.co/"),
    "spaces": ("space", "https://huggingface.co/spaces/"),
    "datasets": ("dataset", "https://huggingface.co/datasets/"),
}
