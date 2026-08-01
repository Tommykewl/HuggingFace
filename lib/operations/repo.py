"""RepoOperation — the shared base of the repo operations (load / unload /
git proxy), which act on the Hub repos tracked as git submodules under
hf/<namespace>/<entity>/<name> (entities: models, spaces, datasets).
Holds the <entity> <service> <namespace> <name> argument validation and
the submodule preconditions they share."""

import sys

from lib.operations.baseoperation import BaseOperation
from lib.config import HF_DIR, REPO_KINDS, ROOT
from lib.utilities import registered_services, registered_submodules


class RepoOperation(BaseOperation):
    """Shared base for the repo operations: the entity/service/namespace/
    name argument validation and the submodule preconditions they share."""

    mandatory_count = 4                    # <entity> <service> <namespace> <name>
    optional_max = None                    # no vargs (GitOperation overrides)

    def _target(self, mandatory):
        """Validate <entity> <service> <namespace> <name>; return (entity,
        service, ns, name, rel, path). The entity must be a repo kind
        (models|spaces|datasets); the service must be registered — and only
        huggingface hosts git repos loadable as submodules, so anything
        else errors here."""
        if len(mandatory) != 4 or not all(mandatory):
            raise ValueError("expected: <models|spaces|datasets> <service> "
                             "<namespace> <name>")
        entity, service, ns, name = mandatory
        if entity not in REPO_KINDS:
            raise ValueError(f"unknown entity '{entity}' — expected: "
                             f"{' | '.join(REPO_KINDS)}")
        services = registered_services()
        if service not in services:
            raise ValueError(f"unknown service '{service}' — registered: "
                             f"{' | '.join(services)}")
        if service != "huggingface":
            raise ValueError(f"service '{service}' hosts no git repos — only "
                             "huggingface repos are tracked as submodules "
                             "under hf/")
        rel = f"hf/{ns}/{entity}/{name}"
        return entity, service, ns, name, rel, ROOT / rel

    @staticmethod
    def _require_tracked_submodule(entity, ns, rel):
        """Unload/git precondition: namespace under hf/, tracked submodule."""
        if not (HF_DIR / ns).is_dir():
            sys.exit(f"ERROR: namespace '{ns}' does not exist under hf/ — "
                     "see: list namespaces")
        if rel not in registered_submodules():
            sys.exit(f"ERROR: '{rel}' is not a tracked submodule — "
                     f"see: list {entity} / list namespaces")
