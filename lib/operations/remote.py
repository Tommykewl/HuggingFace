"""RemoteOperation — the shared base of the remote-entity operations
(create / delete), which act on an entity DIRECTLY on one remote service
(the Hugging Face Hub, Kaggle, ...) rather than on a local checkout.
Holds the <entity> <service> <namespace> <name> argument validation they
share."""

from lib.operations.baseoperation import BaseOperation
from lib.utilities import reachable_services

# The entities the remote operations act on (namespaces are accounts —
# never created or deleted from here). Whether a given SERVICE supports
# a given entity is the service's own call: its _create/_delete error out
# on anything it cannot do (the BaseService contract).
REMOTE_ENTITIES = ("models", "spaces", "datasets", "jobs")


class RemoteOperation(BaseOperation):
    """Shared base for create/delete: the entity/service/namespace/name
    validation."""

    mandatory_count = 4                    # <entity> <service> <namespace> <name>
    optional_max = None                    # no vargs

    def _resolve(self, mandatory, optional, vargs):
        """Validate <entity> <service> <namespace> <name>; return (entity,
        service_name, service, namespace, name). The service must be
        registered AND have credentials in .env — resolved via the same
        reachable_services sweep `list` uses, so the error names every
        usable service."""
        if optional or vargs:
            raise ValueError(f"unexpected arguments: {' '.join((*optional, *vargs))}")
        if len(mandatory) != 4 or not all(mandatory):
            raise ValueError("expected: <models|spaces|datasets|jobs> "
                             "<service> <namespace> <name>")
        entity, service_name, namespace, name = mandatory
        if entity not in REMOTE_ENTITIES:
            raise ValueError(f"unknown entity '{entity}' — expected: "
                             f"{' | '.join(REMOTE_ENTITIES)}")
        services = dict(reachable_services())
        if service_name not in services:
            raise ValueError(f"unknown or unreachable service '{service_name}'"
                             f" — reachable: {' | '.join(services) or '(none)'}")
        return entity, service_name, services[service_name], namespace, name
