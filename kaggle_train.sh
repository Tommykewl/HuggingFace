#!/usr/bin/env bash
#
# kaggle_train.sh — run a model's trainer.kaggle.ipynb on Kaggle's free GPU,
# headless, in a single push.
#
# This is a workspace-level launcher. It takes ONE argument: the model path
# <namespace>/<model name> (the folder holding that model's trainer.kaggle.ipynb),
# e.g.  ./kaggle_train.sh smallTech/rtdetrv2-r50vd-sportsmot-players
# If the argument is omitted it prompts for it, and it verifies that the folder
# and its trainer.kaggle.ipynb exist before doing anything.
#
# The HF token is NOT a Kaggle Secret (those get dropped when the CLI pushes a new
# version). Instead the notebook reads it from a private Kaggle dataset named
# "external-secrets", which we reference in the kernel metadata below — dataset
# references DO persist across versions. Create that dataset once on Kaggle: a
# private dataset "external-secrets" holding a `secrets` file with HF_TOKEN=<...>.
#
# Prerequisites (one-time):
#   1. pip install --user kaggle
#   2. Authenticate the Kaggle CLI (kaggle.json OR ~/.kaggle/access_token);
#      `kaggle config view` should show your username.
#   3. A private "external-secrets" Kaggle dataset containing HF_TOKEN (see above).
#
# Usage:  ./kaggle_train.sh <namespace>/<model name>
set -euo pipefail

# The `kaggle` console script lives in the pip --user bin dir, which may not be
# on PATH; add it so this script works from any shell.
export PATH="$HOME/Library/Python/3.12/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")" && pwd)"
NOTEBOOK="trainer.kaggle.ipynb"
SECRETS_SLUG="external-secrets"

command -v kaggle >/dev/null || { echo "kaggle CLI not found on PATH"; exit 1; }

# Resolve and validate the model argument (<namespace>/<model name>). If not
# passed, ask for it; keep asking until the folder + notebook actually exist.
MODEL="${1:-}"
while true; do
  if [ -z "$MODEL" ]; then
    read -r -p "Enter <namespace>/<model name> (e.g. smallTech/rtdetrv2-r50vd-sportsmot-players): " MODEL
  fi
  DIR="$ROOT/$MODEL"
  if [ -d "$DIR" ] && [ -f "$DIR/$NOTEBOOK" ]; then
    break
  fi
  echo "Not found: expected a folder '$MODEL' containing $NOTEBOOK under $ROOT"
  MODEL=""   # force a re-prompt
done

# Kaggle kernels are owned by a user account, so the id must use YOUR Kaggle
# username (from `kaggle config view`). The kernel slug is the model name.
SLUG="$(basename "$MODEL")"
KAGGLE_USER="$(kaggle config view 2>/dev/null | sed -n 's/^- username: *//p')"
[ -n "$KAGGLE_USER" ] || { echo "Not authenticated — no username from 'kaggle config view'. Check ~/.kaggle/access_token or ~/.kaggle/kaggle.json"; exit 1; }
KERNEL="$KAGGLE_USER/$SLUG"
SECRETS_DS="$KAGGLE_USER/$SECRETS_SLUG"

# The token-bearing dataset must exist, or the run would waste GPU and fall back
# to saving the model to the kernel output instead of pushing to the Hub.
if ! kaggle datasets files "$SECRETS_DS" >/dev/null 2>&1; then
  echo "Secrets dataset '$SECRETS_DS' not found. Create a private Kaggle dataset '$SECRETS_SLUG' with a 'secrets' file containing HF_TOKEN=<...> first."
  exit 1
fi

echo "Model: $MODEL"
echo "Kernel: $KERNEL   secrets: $SECRETS_DS   (HF target: https://huggingface.co/$MODEL)"

# Stage the notebook + generated metadata; reference the secrets dataset so the
# notebook can read HF_TOKEN from /kaggle/input/external-secrets/secrets.
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
cp "$DIR/$NOTEBOOK" "$STAGE/"
cat > "$STAGE/kernel-metadata.json" <<JSON
{
  "id": "$KERNEL",
  "title": "$SLUG",
  "code_file": "$NOTEBOOK",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["$SECRETS_DS"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
JSON

echo "Pushing to Kaggle (this queues a GPU run)..."
kaggle kernels push -p "$STAGE"

cat <<EOF

Training launched. Monitor it:
  kaggle kernels status $KERNEL          # QUEUED / RUNNING / COMPLETE / ERROR
  open https://www.kaggle.com/$KERNEL    # live logs

The notebook reads HF_TOKEN from the $SECRETS_DS dataset and pushes the model to:
  https://huggingface.co/$MODEL

If the run finished but nothing was pushed, the secrets dataset wasn't readable —
the model was saved to the kernel Output tab instead.
EOF
