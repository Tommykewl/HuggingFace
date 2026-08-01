"""`create <entity> <service> <namespace> <name>` — create the entity on
the remote service (a Hub repo on huggingface, ...). The service's _create
decides what it can create and errors out on anything else (BaseService
contract)."""

from lib.operations.remote import RemoteOperation


class CreateOperation(RemoteOperation):
    """`create <entity> <service> <namespace> <name>` — create the entity
    remotely."""

    _helptext = [
        ("create models <service> <namespace> <name>",
         "Create <namespace>/<name> as a new model on <service>\n"
         "(huggingface: a Hub model repo; <namespace> must be one the\n"
         "account can write to). A service that cannot create models —\n"
         "e.g. Kaggle, where repos appear via pushes — errors out saying so.\n"
         "Example: create models huggingface smallTech my-model"),
        ("create spaces <service> <namespace> <name>",
         "Create <namespace>/<name> as a new Space on <service>\n"
         "(huggingface: a Hub Space repo, Gradio SDK). Same rules as\n"
         "`create models`."),
        ("create datasets <service> <namespace> <name>",
         "Create <namespace>/<name> as a new dataset on <service>\n"
         "(huggingface: a Hub dataset repo). Same rules as `create models`."),
        ("create jobs <service> <namespace> <name>",
         "Jobs are not created by name on any current service — they are\n"
         "submitted from a runnable via `execute jobs`; every service\n"
         "errors out pointing there."),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, service_name, service, namespace, name = self._resolve(
            mandatory, optional, vargs)
        created = service.create(entity, namespace, name)
        print(f"Created on {service_name}: {created}")
