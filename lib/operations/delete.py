"""`delete <entity> <service> <name>` — delete the entity on the remote
service. DESTRUCTIVE and remote-only: nothing local is touched (a loaded
submodule of a deleted Hub repo must still be unloaded via `unload`). The
service's _delete decides what it can delete and errors out on anything
else (BaseService contract)."""

from lib.operations.remote import RemoteOperation


class DeleteOperation(RemoteOperation):
    """`delete <entity> <service> <name>` — delete the entity remotely."""

    _helptext = [
        ("delete models <service> <name>",
         "Delete the model <name> on <service> (huggingface: the Hub model\n"
         "repo; kaggle: the Kaggle model — '<owner>/<slug>', a bare name\n"
         "goes to the user's namespace). DESTRUCTIVE and remote-only: a\n"
         "locally loaded submodule is not removed — `unload` it separately.\n"
         "Example: delete models huggingface smallTech/my-model"),
        ("delete spaces <service> <name>",
         "Delete the Space <name> on <service> (huggingface: the Hub Space\n"
         "repo). Same name and error rules as `delete models`."),
        ("delete datasets <service> <name>",
         "Delete the dataset <name> on <service> (huggingface: the Hub\n"
         "dataset repo; kaggle: the Kaggle dataset). Same name and error\n"
         "rules as `delete models`."),
        ("delete jobs <service> <name>",
         "Delete the job <name> on <service>: huggingface cancels the Hub\n"
         "Job (ids from `list jobs` / `execute jobs`); kaggle deletes the\n"
         "kernel ('<user>/<kernel-slug>' — the run id `execute` prints)."),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, service_name, service, name = self._resolve(mandatory,
                                                            optional, vargs)
        deleted = service.delete(entity, name)
        print(f"Deleted on {service_name}: {deleted}")
