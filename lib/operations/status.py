"""`status jobs <run_id> [service]` — one run's status."""

from lib.operations.baseoperation import BaseOperation
from lib.utilities import load_service


class StatusOperation(BaseOperation):
    """`status jobs <run_id> [service]` — one run's status (service defaults
    to huggingface)."""

    mandatory_count = 2                    # <entity> <run_id>
    optional_max = None                    # no vargs; [service] is optional

    _helptext = [
        ("status jobs <run_id> [service]",
         "Status of one run, identified by the id `execute jobs` printed.\n"
         "  <run_id>   Hugging Face: the Job id; Kaggle: <user>/<kernel-slug>\n"
         "  [service]  the service that owns the run — defaults to huggingface\n"
         "Example: status jobs my-job-id | status jobs me/training-index kaggle"),
    ]

    def _run(self, mandatory, optional, vargs):
        if len(mandatory) != 2 or len(optional) > 1 or vargs:
            raise ValueError("expected: jobs <run_id> [service]")
        entity, run_id = mandatory
        if entity != "jobs":
            raise ValueError(f"unknown entity '{entity}' — `status` only "
                             "acts on: jobs")
        service_name = optional[0] if optional else "huggingface"
        status = load_service(service_name).get_status(run_id)
        print(f"[{service_name}] {run_id}: {status}")
