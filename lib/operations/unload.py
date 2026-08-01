"""`unload <entity> <service> <namespace> <name> [-f]` — unload a loaded
submodule.
Default: `git submodule deinit` only — the folder is emptied locally but
stays registered, so load/unload are symmetric and repeatable (refuses on
uncommitted changes). With -f: complete cleanup — deinit, `git rm` the
gitlink and its .gitmodules entry, and delete the cached repo under
.git/modules, leaving the superproject clean for sharing; the Hub repo is
the only copy left, so -f additionally refuses on unpushed commits."""

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
    """`unload <entity> <service> <namespace> <name> [-f]` — deinit the
    submodule (folder emptied, stays registered); -f removes it entirely
    (worktree, gitlink, .gitmodules entry, .git/modules cache)."""

    _helptext = [
        ("unload models <service> <namespace> <model_name> [-f]",
         "Unload the local submodule at hf/<namespace>/models/<model_name>\n"
         "(<service>: only huggingface hosts loadable git repos): checks the\n"
         "namespace exists under hf/ and the model is a tracked submodule,\n"
         "refuses if it has uncommitted changes, then `git submodule\n"
         "deinit`s it — the folder is emptied locally but stays registered,\n"
         "so a later `load models` restores it.\n"
         "With -f: complete cleanup — additionally refuses on unpushed\n"
         "commits (the Hub repo is the only copy left afterwards), then\n"
         "deinits, `git rm`s the gitlink and its .gitmodules entry (both\n"
         "staged — commit to share the removal), and deletes the cached repo\n"
         "under .git/modules. `load models` re-adds it from the Hub.\n"
         "Example: unload models huggingface smallTech rtdetr-sportsmot -f"),
        ("unload spaces <service> <namespace> <space_name> [-f]",
         "Unload the local submodule at hf/<namespace>/spaces/<space_name>.\n"
         "Same checks and behaviour as `unload models`."),
        ("unload datasets <service> <namespace> <dataset_name> [-f]",
         "Unload the local submodule at hf/<namespace>/datasets/<dataset_name>.\n"
         "Same checks and behaviour as `unload models`."),
    ]

    def _run(self, mandatory, optional, vargs):
        if vargs or len(optional) > 1 or (optional and optional[0] != "-f"):
            raise ValueError("unexpected arguments: "
                             f"{' '.join((*optional, *vargs))} — the only "
                             "optional argument is -f (complete cleanup)")
        force = bool(optional)
        entity, service, ns, name, rel, target = self._target(mandatory)
        self._require_tracked_submodule(entity, ns, rel)
        if not (target.is_dir() and any(target.iterdir())):
            sys.exit(f"'{rel}' is not loaded — nothing to do")
        # Never discard local work: deinit -f would drop uncommitted changes.
        dirty = git("-C", str(target), "status", "--porcelain", capture=True)
        if dirty.stdout.strip():
            sys.exit(f"ERROR: '{rel}' has uncommitted changes — commit and push "
                     "them to the Hub before unloading:\n" + dirty.stdout.rstrip())
        if not force:
            git("submodule", "deinit", "-f", rel)
            print(f"Unloaded: {rel} (still registered — `load {entity} "
                  f"{service} {ns} {name}` restores it)")
            return
        # -f: the Hub repo becomes the only remaining copy, so unpushed
        # commits block too.
        unpushed = git("-C", str(target), "log", "--oneline", "--branches",
                       "--not", "--remotes", capture=True)
        if unpushed.stdout.strip():
            sys.exit(f"ERROR: '{rel}' has commits not pushed to the Hub — push "
                     "them before unloading with -f:\n" + unpushed.stdout.rstrip())
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
              f"`load {entity} {service} {ns} {name}` re-adds it from the Hub)")
