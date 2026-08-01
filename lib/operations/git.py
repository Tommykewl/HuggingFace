"""`git <entity> <service> <namespace> <name> <git args...>` — proxy a git
command into a loaded submodule verbatim. Git repos exist only for
huggingface repo entities (models/spaces/datasets — tracked as submodules
under hf/), so this operation validates those bounds itself; it holds no
service-side logic to delegate."""

import sys

from lib.operations.baseoperation import BaseOperation
from lib.config import REPO_KINDS, ROOT
from lib.utilities import git, reachable_services, registered_submodules


class GitOperation(BaseOperation):
    """`git <entity> <service> <namespace> <name> <git args...>` — proxy a
    git command to a loaded submodule: everything after <name> runs
    verbatim as `git -C hf/<ns>/<entity>/<name> <git args...>`, and the
    exit code is git's own — output, prompts and errors are git's."""

    mandatory_count = 4                    # <entity> <service> <namespace> <name>
    optional_max = 0                       # every trailing argument is a varg

    _helptext = [
        ("git models <service> <namespace> <model_name> <git args...>",
         "Proxy a git command to the loaded submodule (<service>: only\n"
         "huggingface hosts loadable git repos): everything after\n"
         "<model_name> is passed AS IS to git inside the submodule\n"
         "(`git -C hf/<namespace>/models/<model_name> <git args...>`), and the\n"
         "exit code is git's. The submodule IS the Hub repo, so this is how\n"
         "you inspect/commit/push it without cd-ing around.\n"
         "Example: git models huggingface smallTech rtdetr-sportsmot status\n"
         "         git models huggingface smallTech rtdetr-sportsmot commit "
         "-m \"some commit\""),
        ("git spaces <service> <namespace> <space_name> <git args...>",
         "Proxy a git command to the loaded submodule: runs\n"
         "`git -C hf/<namespace>/spaces/<space_name> <git args...>` verbatim.\n"
         "Example: git spaces huggingface Tamoghna1995 rtdetr-sportsmot "
         "log --oneline -3"),
        ("git datasets <service> <namespace> <dataset_name> <git args...>",
         "Proxy a git command to the loaded submodule: runs\n"
         "`git -C hf/<namespace>/datasets/<dataset_name> <git args...>` verbatim."),
    ]

    def _run(self, mandatory, optional, vargs):
        if len(mandatory) != 4 or not all(mandatory):
            raise ValueError("expected: <models|spaces|datasets> <service> "
                             "<namespace> <name> <git args...>")
        entity, service, ns, name = mandatory
        if entity not in REPO_KINDS:
            raise ValueError(f"unknown entity '{entity}' — expected: "
                             f"{' | '.join(REPO_KINDS)}")
        services = [name_ for name_, _ in reachable_services()]
        if service not in services:
            raise ValueError(f"unknown or unreachable service '{service}' — "
                             f"reachable: {' | '.join(services) or '(none)'}")
        if service != "huggingface":
            raise ValueError(f"service '{service}' hosts no git repos — only "
                             "huggingface repos are tracked as submodules "
                             "under hf/")
        if not vargs:
            raise ValueError("expected: <models|spaces|datasets> <service> "
                             "<namespace> <name> <git args...>")
        rel = f"hf/{ns}/{entity}/{name}"
        target = ROOT / rel
        if rel not in registered_submodules():
            sys.exit(f"ERROR: '{rel}' is not a tracked submodule — "
                     f"see: list {entity}")
        if not (target.is_dir() and any(target.iterdir())):
            sys.exit(f"ERROR: '{rel}' is not loaded — load it first: "
                     f"load {entity} {service} {ns} {name}")
        sys.exit(git("-C", str(target), *vargs, check=False).returncode)
