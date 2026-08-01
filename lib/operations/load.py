"""`load <entity> <service> <namespace> <name>` — materialize the entity's
content locally from the remote service. The HOW is the service's call
(BaseService._load): huggingface repo entities become git submodules under
hf/, kaggle entities are plain downloads, jobs sync from their staging
storage (the source of truth, created by `create jobs`)."""

from lib.operations.remote import RemoteOperation


class LoadOperation(RemoteOperation):
    """`load <entity> <service> <namespace> <name>` — materialize the
    entity locally."""

    _helptext = [
        ("load models <service> <namespace> <model_name>",
         "Load the model <namespace>/<model_name> of <service> locally.\n"
         "huggingface: a git submodule at hf/<namespace>/models/<model_name>\n"
         "(namespace and Hub existence checked; GIT_LFS_SKIP_SMUDGE=1, so\n"
         "weights stay as LFS pointers). kaggle: models cannot be loaded by\n"
         "name (a download needs framework/instance/version).\n"
         "Example: load models huggingface smallTech rtdetr-sportsmot"),
        ("load spaces <service> <namespace> <space_name>",
         "Load the Space <namespace>/<space_name> of <service> locally\n"
         "(huggingface only: a git submodule at\n"
         "hf/<namespace>/spaces/<space_name> — same checks as `load models`).\n"
         "Example: load spaces huggingface Tamoghna1995 rtdetr-sportsmot"),
        ("load datasets <service> <namespace> <dataset_name>",
         "Load the dataset <namespace>/<dataset_name> of <service> locally.\n"
         "huggingface: a git submodule at hf/<namespace>/datasets/<name>\n"
         "(LFS data stays pointers). kaggle: downloaded into the gitignored\n"
         "kaggle/<namespace>/datasets/<name>."),
        ("load jobs <service> <namespace> <job_name>",
         "Load the job's artifacts from its staging storage — the source of\n"
         "truth, created by `create jobs` (missing staging = error).\n"
         "huggingface: syncs the bucket <namespace>/<job_name> into\n"
         "hf/<namespace>/jobs/<job_name>. kaggle: downloads the job's\n"
         "staging dataset into kaggle/<namespace>/jobs/<job_name> (both\n"
         "destinations gitignored — remote-backed caches).\n"
         "Example: load jobs kaggle tamobiswas rtdetr-sportsmot"),
    ]

    def _run(self, mandatory, optional, vargs):
        entity, _, service, namespace, name = self._resolve(
            mandatory, optional, vargs)
        print(f"Loaded: {service.load(entity, namespace, name)}")
