"""`execute jobs <namespace>/<model>/<type>/<script_name>` — resolve a
runnable and submit it to the service that hosts it."""

import json
import sys

from lib.operations.baseoperation import BaseOperation
from lib.config import ROOT, TYPES
from lib.utilities import load_service

# ---------------------------------------------------------------------------
# Script resolution.
# Search order: the model folder's own <type>/ (first-party scripts, run as
# Hugging Face Jobs by default), then every external referenced by the model's
# externals.json. An external's used_for is a list of operation objects:
#   {"name": "<operation>", "artifacts": [{"name": ..., "type": "script" |
#   "result"}, ...]}
# — an external is searched only when it declares the requested operation AND
# the requested name as a 'script' artifact. The declarations are the whole
# search space: a script on disk that isn't declared is simply invisible to
# resolution (it does not exist as far as the launcher is concerned).
# 'result' artifacts are operation OUTPUTS living Hub-side in the model folder
# (<model>/.eval_results/<operation>.yaml) — never runnables. A runnable
# matches when its filename starts with "<script_name>." and has a script
# extension; *.config.json and data files (e.g. results.yaml) never do.
# Exactly one match must remain.
# ---------------------------------------------------------------------------
RUNNABLE_SUFFIXES = (".py", ".ipynb")


def parse_target(target):
    """Validate and split a <namespace>/<model>/<type>/<script_name> target."""
    parts = target.split("/")
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"invalid target '{target}' — expected exactly "
                         "<namespace>/<model>/<type>/<script_name>")
    ns, model_name, type_, name = parts
    if type_ not in TYPES:
        raise ValueError(f"invalid <type> '{type_}' — must be one of: "
                         f"{' | '.join(TYPES)}")
    return ns, model_name, type_, name


def resolve_script(ns, model_name, type_, name):
    """Return (service, script_path). service 'huggingface' == first-party.

    Model folders live at hf/<ns>/models/<model> — git submodules of the HF
    model repos — while targets keep the <ns>/<model>/... syntax.
    """
    model_dir = ROOT / "hf" / ns / "models" / model_name
    if not model_dir.is_dir():
        sys.exit(f"ERROR: model folder not found: hf/{ns}/models/{model_name} "
                 "(a git submodule — load it first: "
                 f"load models huggingface {ns} {model_name})")

    def runnables(directory):
        if not directory.is_dir():
            return []
        return [p for p in sorted(directory.iterdir())
                if p.is_file() and p.name.startswith(name + ".")
                and p.suffix in RUNNABLE_SUFFIXES
                and not p.name.endswith(".config.json")]

    candidates = [("huggingface", p) for p in runnables(model_dir / type_)]
    externals = model_dir / "externals.json"
    if externals.is_file():
        for entry in json.load(open(externals)).get("externals", []):
            service, ext_path = entry.get("service"), entry.get("path")
            if not service or not ext_path:
                continue
            operation = next(
                (op for op in entry.get("used_for", [])
                 if isinstance(op, dict) and op.get("name") == type_), None)
            if operation is None:
                continue               # external not used for this operation
            declared = {a.get("name") for a in operation.get("artifacts", [])
                        if a.get("type") == "script"}
            if name not in declared:
                continue               # undeclared -> invisible to resolution
            # The entry's path IS the external's root — by convention
            # <service>/<namespace>/jobs/<reference> (the namespace being
            # the account name on that service, mirroring hf/<ns>/...).
            for p in runnables(ROOT / ext_path / type_):
                candidates.append((service, p))

    if not candidates:
        sys.exit(f"ERROR[no-script-found]: no runnable '{name}.*' under "
                 f"{ns}/{model_name}/{type_} or any external it references")
    if len(candidates) > 1:
        listing = "\n  ".join(f"{s}: {p.relative_to(ROOT)}" for s, p in candidates)
        sys.exit("ERROR[ambiguous]: multiple runnables match "
                 f"'{name}' — rename so exactly one matches:\n  {listing}")
    service, path = candidates[0]
    print(f"Resolved: service={service}  script={path.relative_to(ROOT)}")
    return service, path


class ExecuteOperation(BaseOperation):
    """`execute jobs <namespace>/<model>/<type>/<script_name>` — resolve the
    runnable and submit it to the service that hosts it."""

    mandatory_count = 2                    # <entity> <target>
    optional_max = None                    # no vargs

    _helptext = [
        ("execute jobs <namespace>/<model>/<type>/<script_name>",
         "Submit one of a model's runnables to the service that hosts it.\n"
         "  <namespace>    Hugging Face namespace, e.g. smallTech\n"
         "  <model>        model folder = Hub repo id, e.g. rtdetr-sportsmot\n"
         f"  <type>         one of: {' | '.join(TYPES)}\n"
         "  <script_name>  runnable's base name, e.g. index, smoketest, prepare-data\n"
         "The script is resolved in hf/<namespace>/models/<model>/<type>/ first\n"
         "(first-party -> Hugging Face Jobs), then in every external service the\n"
         "model's externals.json declares for that operation; zero or multiple\n"
         "matches are errors. Prints a run id for `status jobs`. A run whose\n"
         "script resolved from a job folder (<service>/<ns>/jobs/<job>/...)\n"
         "is recorded into that job's staging: .runs/<run_id>/ gets the\n"
         "script + config snapshot (output.log follows via `status jobs`).\n"
         "Example: execute jobs smallTech/rtdetr-sportsmot/training/smoketest"),
    ]

    def _run(self, mandatory, optional, vargs):
        if len(mandatory) != 2 or optional or vargs:
            raise ValueError("expected: jobs "
                             "<namespace>/<model>/<type>/<script_name>")
        entity, target = mandatory
        if entity != "jobs":
            raise ValueError(f"unknown entity '{entity}' — `execute` only "
                             "acts on: jobs")
        ns, model_name, type_, name = parse_target(target)
        service, script = resolve_script(ns, model_name, type_, name)
        svc = load_service(service)
        run_id = svc.run(script, f"{ns}/{model_name}", type_, name)
        print(f"run id: {run_id}   (use it with: status jobs {run_id} {service})")
        # Job-resolved runs are recorded into the job's staging storage:
        # .runs/<run_id>/ gets a snapshot of the executed script + config
        # (output.log follows via `status jobs` once the run is terminal).
        # A script path <service>/<namespace>/jobs/<job>/... IS the job
        # membership; first-party scripts (model repo) belong to no job.
        parts = script.relative_to(ROOT).parts
        if len(parts) > 3 and parts[2] == "jobs":
            print(svc.record_run(parts[1], parts[3], run_id, script))
