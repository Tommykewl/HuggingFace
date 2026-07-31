"""RepoOperation — the shared base of the repo operations (load / unload /
git proxy), which act on the Hub repos tracked as git submodules under
hf/<namespace>/<entity>/<name> (entities: models, spaces, datasets).
Holds the <entity> <namespace> <name> argument validation and the
submodule preconditions they share."""

import sys

from lib.operations.baseoperation import BaseOperation
from lib.config import HF_DIR, REPO_KINDS, ROOT
from lib.utilities import registered_submodules


class RepoOperation(BaseOperation):
    """Shared base for the repo operations: the entity/namespace/name
    argument validation and the submodule preconditions they share."""

    mandatory_count = 3                    # <entity> <namespace> <name>
    optional_max = None                    # no vargs (GitOperation overrides)

    def _target(self, mandatory):
        """Validate <entity> <namespace> <name>; return (entity, ns, name,
        rel, path). The entity must be a repo kind (models|spaces|datasets)."""
        if len(mandatory) != 3 or not all(mandatory):
            raise ValueError("expected: <models|spaces|datasets> "
                             "<namespace> <name>")
        entity, ns, name = mandatory
        if entity not in REPO_KINDS:
            raise ValueError(f"unknown entity '{entity}' — expected: "
                             f"{' | '.join(REPO_KINDS)}")
        rel = f"hf/{ns}/{entity}/{name}"
        return entity, ns, name, rel, ROOT / rel

    @staticmethod
    def _require_tracked_submodule(entity, ns, rel):
        """Unload/git precondition: namespace under hf/, tracked submodule."""
        if not (HF_DIR / ns).is_dir():
            sys.exit(f"ERROR: namespace '{ns}' does not exist under hf/ — "
                     "see: list namespaces")
        if rel not in registered_submodules():
            sys.exit(f"ERROR: '{rel}' is not a tracked submodule — "
                     f"see: list {entity} / list namespaces")
