"""`load <entity> <service> <namespace> <name>` — materialize a Hub repo
locally as a git submodule under hf/. Validates the service (only
huggingface hosts loadable git repos), the namespace (one of the account's,
per `list namespaces`) and the repo's existence on the Hub, then `git
submodule add`s it (or `update --init`s it when already registered from an
earlier add) with GIT_LFS_SKIP_SMUDGE=1 so LFS payloads stay pointers."""

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


def check_account_namespace(service, ns):
    """Check 1: <namespace> must be one of the account's namespaces on the
    given service."""
    namespaces = load_service(service).list("namespaces")
    if ns not in namespaces:
        sys.exit(f"ERROR: '{ns}' is not a namespace of this account on "
                 f"{service} ({', '.join(namespaces)}) — see: list namespaces")


class LoadOperation(RepoOperation):
    """`load <entity> <service> <namespace> <name>` — materialize the Hub
    repo as a git submodule under hf/."""

    _helptext = [
        ("load models <service> <namespace> <model_name>",
         "Load the model repo <namespace>/<model_name> of <service> locally\n"
         "as a git submodule at hf/<namespace>/models/<model_name>. Only\n"
         "huggingface hosts loadable git repos (other services error out).\n"
         "Checks that <namespace> is one of the account's namespaces (see\n"
         "`list namespaces`) and that the model exists on the Hub, then `git\n"
         "submodule add`s it — or `git submodule update --init`s it when\n"
         "already registered — with GIT_LFS_SKIP_SMUDGE=1, so weights stay as\n"
         "LFS pointers.\n"
         "Example: load models huggingface smallTech rtdetr-sportsmot"),
        ("load spaces <service> <namespace> <space_name>",
         "Load the Space repo <namespace>/<space_name> of <service> locally\n"
         "as a git submodule at hf/<namespace>/spaces/<space_name>. Same\n"
         "checks and submodule setup as `load models`.\n"
         "Example: load spaces huggingface Tamoghna1995 rtdetr-sportsmot"),
        ("load datasets <service> <namespace> <dataset_name>",
         "Load the dataset repo <namespace>/<dataset_name> of <service>\n"
         "locally as a git submodule at hf/<namespace>/datasets/<dataset_name>.\n"
         "Same checks and submodule setup as `load models` (LFS data stays\n"
         "as pointers)."),
    ]

    def _run(self, mandatory, optional, vargs):
        if optional or vargs:
            raise ValueError(f"unexpected arguments: {' '.join((*optional, *vargs))}")
        entity, service, ns, name, rel, target = self._target(mandatory)
        repo_type, url_prefix = REPO_KINDS[entity]
        check_account_namespace(service, ns)
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
