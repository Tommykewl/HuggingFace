"""`unload <entity> <namespace> <name>` — deinit a loaded submodule: its
folder is emptied locally but it stays registered, so load/unload are
symmetric and repeatable. Refuses on uncommitted changes."""

import sys

from lib.utilities import git
from lib.operations.repo import RepoOperation


class UnloadOperation(RepoOperation):
    """`unload <entity> <namespace> <name>` — deinit the submodule (its
    folder is emptied locally; it stays registered)."""

    _helptext = [
        ("unload models <namespace> <model_name>",
         "Unload the local submodule at hf/<namespace>/models/<model_name>:\n"
         "checks the namespace exists under hf/ and the model is a tracked\n"
         "submodule, refuses if it has uncommitted changes (push them to the\n"
         "Hub first), then `git submodule deinit`s it — the folder is emptied\n"
         "locally but stays registered, so a later `load models` restores it.\n"
         "Example: unload models smallTech rtdetr-sportsmot"),
        ("unload spaces <namespace> <space_name>",
         "Unload the local submodule at hf/<namespace>/spaces/<space_name>.\n"
         "Same checks and deinit behaviour as `unload models`."),
        ("unload datasets <namespace> <dataset_name>",
         "Unload the local submodule at hf/<namespace>/datasets/<dataset_name>.\n"
         "Same checks and deinit behaviour as `unload models`."),
    ]

    def _run(self, mandatory, optional, vargs):
        if optional or vargs:
            raise ValueError(f"unexpected arguments: {' '.join((*optional, *vargs))}")
        entity, ns, name, rel, target = self._target(mandatory)
        self._require_tracked_submodule(entity, ns, rel)
        if not (target.is_dir() and any(target.iterdir())):
            sys.exit(f"'{rel}' is not loaded — nothing to do")
        # Never discard local work: deinit -f would drop uncommitted changes.
        dirty = git("-C", str(target), "status", "--porcelain", capture=True)
        if dirty.stdout.strip():
            sys.exit(f"ERROR: '{rel}' has uncommitted changes — commit and push "
                     "them to the Hub before unloading:\n" + dirty.stdout.rstrip())
        git("submodule", "deinit", "-f", rel)
        print(f"Unloaded: {rel} (still registered — `load {entity} {ns} {name}` "
              "restores it)")
