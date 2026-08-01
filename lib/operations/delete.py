"""`delete <entity> <service> <namespace> <name>` — delete the entity on
the remote service. DESTRUCTIVE and remote-only: nothing local is touched
(a loaded submodule of a deleted Hub repo must still be unloaded via
`unload`). The service's _delete decides what it can delete and errors out
on anything else (BaseService contract)."""

from lib.operations.remote import RemoteOperation


class DeleteOperation(RemoteOperation):
    """`delete <entity> <service> <namespace> <name>` — delete the entity
    remotely."""

    _helptext = [
        ("delete models <service> <namespace> <name>",
         "Delete the model <namespace>/<name> on <service> (huggingface:\n"
         "the Hub model repo; kaggle: the Kaggle model). DESTRUCTIVE and\n"
         "remote-only: a locally loaded submodule is not removed — `unload`\n"
         "it separately.\n"
         "Example: delete models huggingface smallTech my-model"),
        ("delete spaces <service> <namespace> <name>",
         "Delete the Space <namespace>/<name> on <service> (huggingface:\n"
         "the Hub Space repo). Same rules as `delete models`."),
        ("delete datasets <service> <namespace> <name>",
         "Delete the dataset <namespace>/<name> on <service> (huggingface:\n"
         "the Hub dataset repo; kaggle: the Kaggle dataset). Same rules as\n"
         "`delete models`."),
        ("delete jobs <service> <namespace> <name>",
         "Delete the job <name> in <namespace> on <service>: huggingface\n"
         "cancels the Hub Job (<name> = the job id from `list jobs` /\n"
         "`execute jobs`, <namespace> = the account it ran under); kaggle\n"
         "deletes the kernel (<name> = the kernel slug, <namespace> = its\n"
         "owner)."),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, service_name, service, namespace, name = self._resolve(
            mandatory, optional, vargs)
        deleted = service.delete(entity, namespace, name)
        print(f"Deleted on {service_name}: {deleted}")
