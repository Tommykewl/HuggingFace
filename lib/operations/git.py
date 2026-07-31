"""`git <entity> <namespace> <name> <git args...>` — proxy a git command
into a loaded submodule verbatim."""

import sys

from lib.utilities import git
from lib.operations.repo import RepoOperation


class GitOperation(RepoOperation):
    """`git <entity> <namespace> <name> <git args...>` — proxy a git command
    to a loaded submodule: everything after <name> runs verbatim as
    `git -C hf/<ns>/<entity>/<name> <git args...>`, and the exit code is
    git's own — output, prompts and errors are git's."""

    optional_max = 0                       # every trailing argument is a varg

    _helptext = [
        ("git models <namespace> <model_name> <git args...>",
         "Proxy a git command to the loaded submodule: everything after\n"
         "<model_name> is passed AS IS to git inside the submodule\n"
         "(`git -C hf/<namespace>/models/<model_name> <git args...>`), and the\n"
         "exit code is git's. The submodule IS the Hub repo, so this is how\n"
         "you inspect/commit/push it without cd-ing around.\n"
         "Example: git models smallTech rtdetr-sportsmot status\n"
         "         git models smallTech rtdetr-sportsmot commit -m \"some commit\""),
        ("git spaces <namespace> <space_name> <git args...>",
         "Proxy a git command to the loaded submodule: runs\n"
         "`git -C hf/<namespace>/spaces/<space_name> <git args...>` verbatim.\n"
         "Example: git spaces Tamoghna1995 rtdetr-sportsmot log --oneline -3"),
        ("git datasets <namespace> <dataset_name> <git args...>",
         "Proxy a git command to the loaded submodule: runs\n"
         "`git -C hf/<namespace>/datasets/<dataset_name> <git args...>` verbatim."),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, ns, name, rel, target = self._target(mandatory)
        if not vargs:
            raise ValueError("expected: <models|spaces|datasets> <namespace> "
                             "<name> <git args...>")
        self._require_tracked_submodule(entity, ns, rel)
        if not (target.is_dir() and any(target.iterdir())):
            sys.exit(f"ERROR: '{rel}' is not loaded — load it first: "
                     f"load {entity} {ns} {name}")
        sys.exit(git("-C", str(target), *vargs, check=False).returncode)
