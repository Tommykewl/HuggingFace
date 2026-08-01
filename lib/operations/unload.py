"""`unload <entity> <namespace> <name>` — fully remove a loaded submodule:
deinit, `git rm` the gitlink and its .gitmodules entry, and delete the
cached repo under .git/modules, leaving the superproject clean for sharing.
Refuses on uncommitted or unpushed work (the Hub repo is the only copy
left afterwards). A later `load` re-adds it from the Hub."""

import shutil
import sys

from lib.config import ROOT
from lib.utilities import git
from lib.operations.repo import RepoOperation


def submodule_name(rel):
    """The .gitmodules section name whose path is rel (name and path can
    differ — `submodule add` derives the name, and moves may desync them)."""
    result = git("config", "-f", ".gitmodules", "--get-regexp",
                 r"submodule\..*\.path", capture=True, check=False)
    for line in result.stdout.splitlines():
        key, _, path = line.partition(" ")
        if path == rel:
            return key[len("submodule."):-len(".path")]
    return None


class UnloadOperation(RepoOperation):
    """`unload <entity> <namespace> <name>` — remove the submodule entirely
    (worktree, gitlink, .gitmodules entry, .git/modules cache)."""

    _helptext = [
        ("unload models <namespace> <model_name>",
         "Remove the submodule at hf/<namespace>/models/<model_name> entirely:\n"
         "checks the namespace exists under hf/ and the model is a tracked\n"
         "submodule, refuses if it has uncommitted or unpushed work (the Hub\n"
         "repo is the only copy left afterwards), then deinits it, `git rm`s\n"
         "the gitlink and its .gitmodules entry (both staged — commit to\n"
         "share the removal), and deletes the cached repo under .git/modules.\n"
         "A later `load models` re-adds it from the Hub.\n"
         "Example: unload models smallTech rtdetr-sportsmot"),
        ("unload spaces <namespace> <space_name>",
         "Remove the submodule at hf/<namespace>/spaces/<space_name> entirely.\n"
         "Same checks and removal behaviour as `unload models`."),
        ("unload datasets <namespace> <dataset_name>",
         "Remove the submodule at hf/<namespace>/datasets/<dataset_name>\n"
         "entirely. Same checks and removal behaviour as `unload models`."),
    ]

    def _run(self, mandatory, optional, vargs):
        if optional or vargs:
            raise ValueError(f"unexpected arguments: {' '.join((*optional, *vargs))}")
        entity, ns, name, rel, target = self._target(mandatory)
        self._require_tracked_submodule(entity, ns, rel)
        if not (target.is_dir() and any(target.iterdir())):
            sys.exit(f"'{rel}' is not loaded — nothing to do")
        # Never discard local work: after removal the Hub repo is the only
        # remaining copy, so uncommitted AND unpushed changes both block.
        dirty = git("-C", str(target), "status", "--porcelain", capture=True)
        if dirty.stdout.strip():
            sys.exit(f"ERROR: '{rel}' has uncommitted changes — commit and push "
                     "them to the Hub before unloading:\n" + dirty.stdout.rstrip())
        unpushed = git("-C", str(target), "log", "--oneline", "--branches",
                       "--not", "--remotes", capture=True)
        if unpushed.stdout.strip():
            sys.exit(f"ERROR: '{rel}' has commits not pushed to the Hub — push "
                     "them before unloading:\n" + unpushed.stdout.rstrip())
        # Resolve the .git/modules cache path BEFORE git rm drops the
        # .gitmodules entry the name comes from.
        modname = submodule_name(rel)
        git("submodule", "deinit", "-f", rel)
        # git rm removes the worktree, the gitlink, and the .gitmodules
        # entry, staging all of it.
        git("rm", "-f", rel)
        if modname:
            shutil.rmtree(ROOT / ".git" / "modules" / modname,
                          ignore_errors=True)
        print(f"Removed: {rel} (staged — commit to share the removal; "
              f"`load {entity} {ns} {name}` re-adds it from the Hub)")
