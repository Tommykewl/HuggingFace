"""`unload <entity> <service> <namespace> <name> [-f]` — remove the
entity's local copy; the remote keeps the content. The HOW is the
service's call (BaseService._unload): huggingface repo submodules deinit
(or fully remove with -f), everything else is a plain local delete."""

from lib.operations.remote import RemoteOperation


class UnloadOperation(RemoteOperation):
    """`unload <entity> <service> <namespace> <name> [-f]` — remove the
    entity's local copy."""

    _helptext = [
        ("unload models <service> <namespace> <model_name> [-f]",
         "Unload the local copy of model <namespace>/<model_name>.\n"
         "huggingface: `git submodule deinit` — the folder is emptied but\n"
         "stays registered, so `load models` restores it; refuses on\n"
         "uncommitted changes. With -f: complete removal — additionally\n"
         "refuses on unpushed commits (the Hub repo is the only copy left),\n"
         "then `git rm`s the gitlink and its .gitmodules entry (staged —\n"
         "commit to share) and deletes the .git/modules cache.\n"
         "Example: unload models huggingface smallTech rtdetr-sportsmot -f"),
        ("unload spaces <service> <namespace> <space_name> [-f]",
         "Unload the local copy of Space <namespace>/<space_name>\n"
         "(huggingface only). Same behaviour as `unload models`."),
        ("unload datasets <service> <namespace> <dataset_name> [-f]",
         "Unload the local copy of dataset <namespace>/<dataset_name>.\n"
         "huggingface: as `unload models`. kaggle: deletes the downloaded\n"
         "folder kaggle/<namespace>/datasets/<name> (-f is meaningless and\n"
         "ignored — nothing is git-registered)."),
        ("unload jobs <service> <namespace> <job_name>",
         "Delete the job's local artifacts folder (hf/<ns>/jobs/<name> or\n"
         "kaggle/<ns>/jobs/<name>) — the staging storage keeps the\n"
         "artifacts: it is the source of truth, `load jobs` restores them."),
    ]

    def _run(self, mandatory, optional, vargs):
        if vargs or len(optional) > 1 or (optional and optional[0] != "-f"):
            raise ValueError("unexpected arguments: "
                             f"{' '.join((*optional, *vargs))} — the only "
                             "optional argument is -f (complete cleanup)")
        force = bool(optional)
        entity, _, service, namespace, name = self._resolve(
            mandatory, [], vargs)
        print(f"Unloaded: {service.unload(entity, namespace, name, force)}")
