"""`load <entity> <namespace> <name>` — materialize a Hub repo locally as a
git submodule under hf/. Validates the namespace (one of the account's, per
`list namespaces`) and the repo's existence on the Hub, then `git submodule
add`s it (or `update --init`s it when already registered from an earlier
add) with GIT_LFS_SKIP_SMUDGE=1 so LFS payloads stay pointers."""

import sys

from lib.config import REPO_KINDS
from lib.utilities import git, load_service, registered_submodules
from lib.operations.repo import RepoOperation


def hub():
    """The huggingface_hub module (installed by the launcher's uv sync)."""
    try:
        import huggingface_hub
    except ImportError:
        sys.exit("The huggingface_hub library is required (the hf CLI is only a "
                 "wrapper around it). It is installed by the launcher's `uv sync` "
                 "— run via ./mlops.sh, or: uv sync")
    return huggingface_hub


def check_account_namespace(ns):
    """Check 1: <namespace> must be one of the account's namespaces."""
    namespaces = load_service("huggingface").list("namespaces")
    if ns not in namespaces:
        sys.exit(f"ERROR: '{ns}' is not a namespace of this account "
                 f"({', '.join(namespaces)}) — see: list namespaces")


class LoadOperation(RepoOperation):
    """`load <entity> <namespace> <name>` — materialize the Hub repo as a
    git submodule under hf/."""

    _helptext = [
        ("load models <namespace> <model_name>",
         "Load the Hub model repo <namespace>/<model_name> locally as a git\n"
         "submodule at hf/<namespace>/models/<model_name>. Checks that\n"
         "<namespace> is one of the account's namespaces (see `list\n"
         "namespaces`) and that the model exists on the Hub, then `git\n"
         "submodule add`s it — or `git submodule update --init`s it when\n"
         "already registered — with GIT_LFS_SKIP_SMUDGE=1, so weights stay as\n"
         "LFS pointers.\n"
         "Example: load models smallTech rtdetr-sportsmot"),
        ("load spaces <namespace> <space_name>",
         "Load the Hub Space repo <namespace>/<space_name> locally as a git\n"
         "submodule at hf/<namespace>/spaces/<space_name>. Same checks and\n"
         "submodule setup as `load models`.\n"
         "Example: load spaces Tamoghna1995 rtdetr-sportsmot"),
        ("load datasets <namespace> <dataset_name>",
         "Load the Hub dataset repo <namespace>/<dataset_name> locally as a git\n"
         "submodule at hf/<namespace>/datasets/<dataset_name>. Same checks and\n"
         "submodule setup as `load models` (LFS data stays as pointers)."),
    ]

    def _run(self, mandatory, optional, vargs):
        if optional or vargs:
            raise ValueError(f"unexpected arguments: {' '.join((*optional, *vargs))}")
        entity, ns, name, rel, target = self._target(mandatory)
        repo_type, url_prefix = REPO_KINDS[entity]
        check_account_namespace(ns)
        # Check 2: the repo must exist on the Hub before anything is cloned.
        if not hub().HfApi().repo_exists(f"{ns}/{name}", repo_type=repo_type):
            sys.exit(f"ERROR: {repo_type} '{ns}/{name}' does not exist on the "
                     f"Hugging Face Hub — see: list {entity}")
        if target.is_dir() and any(target.iterdir()):
            sys.exit(f"'{rel}' is already loaded — nothing to do")
        url = f"{url_prefix}{ns}/{name}"
        if rel in registered_submodules():
            # Registered in .gitmodules — just init it.
            git("submodule", "update", "--init", rel, lfs_skip=True)
        else:
            git("submodule", "add", url, rel, lfs_skip=True)
            print("note: the new submodule is staged (.gitmodules + gitlink) — "
                  "commit to keep it tracked")
        print(f"Loaded: {rel}  <->  {url}")
