"""`create <entity> <service> <name>` — create the entity on the remote
service (a Hub repo on huggingface, ...). The service's _create decides
what it can create and errors out on anything else (BaseService contract)."""

from lib.operations.remote import RemoteOperation


class CreateOperation(RemoteOperation):
    """`create <entity> <service> <name>` — create the entity remotely."""

    _helptext = [
        ("create models <service> <name>",
         "Create <name> as a new model on <service> (huggingface: a Hub\n"
         "model repo; <name> may be '<namespace>/<name>' — a bare name goes\n"
         "to the user's namespace). A service that cannot create models —\n"
         "e.g. Kaggle, where repos appear via pushes — errors out saying so.\n"
         "Example: create models huggingface smallTech/my-model"),
        ("create spaces <service> <name>",
         "Create <name> as a new Space on <service> (huggingface: a Hub\n"
         "Space repo, Gradio SDK). Same name and error rules as\n"
         "`create models`."),
        ("create datasets <service> <name>",
         "Create <name> as a new dataset on <service> (huggingface: a Hub\n"
         "dataset repo). Same name and error rules as `create models`."),
        ("create jobs <service> <name>",
         "Jobs are not created by name on any current service — they are\n"
         "submitted from a runnable via `execute jobs`; every service\n"
         "errors out pointing there."),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, service_name, service, name = self._resolve(mandatory,
                                                            optional, vargs)
        created = service.create(entity, name)
        print(f"Created on {service_name}: {created}")
