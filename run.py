#!/usr/bin/env python3
"""run.py — the workspace runner (steps 2–4). Invoked by run.sh / run.ps1
after they complete step 1 (prerequisite checks); can also be called directly:

    python3 run.py <namespace>/<model>/<type>/<script_name>

Steps hosted here, one function each:
  2. ask_target      — take the target from argv or prompt for it, and
                       validate the exact <namespace>/<model>/<type>/<name>
                       shape (<type> is one of the three stage names).
  3. resolve_script  — find the runnable in the model folder or, via the
                       model's externals.json, under external/<service>/...;
                       error on zero matches (no-script-found) and on more
                       than one (ambiguity, listing all candidates).
  4. run             — every service is a class implementing the
                       BaseService interface (workspace baseservice.py).
                       External services live in external/<service>/service.py
                       (class only — no main), which run.py imports
                       dynamically and calls in-process: no subprocess.
                       First-party scripts (in the model folder itself) use
                       the built-in HuggingFaceService below, which submits
                       Hugging Face Jobs via the huggingface_hub library —
                       the hf CLI is just a wrapper around it, so no
                       service.py exists for huggingface.
"""

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # so service modules can import baseservice
TYPES = ("data-preparation", "training", "evaluation", "testing")

from dotenv import load_dotenv         # noqa: E402
from baseservice import BaseService    # noqa: E402  (needs ROOT on sys.path)

# The workspace .env (template: .env.example) — credentials the services'
# login flows resolve from the environment. Loaded once here at startup;
# run.py is invoked from the workspace root, where .env lives.
load_dotenv()


# ---------------------------------------------------------------------------
# Step 2 — ask the user for the target (or take it from argv).
# ---------------------------------------------------------------------------
def ask_target(argv):
    """Return (namespace, model_name, type, script_name) from argv or a prompt."""
    target = argv[1] if len(argv) > 1 and argv[1] else ""
    if not target:
        target = input(
            "Enter <namespace>/<model>/<type>/<script_name> "
            "(e.g. smallTech/rtdetr-sportsmot/training/index): "
        ).strip()
    parts = target.split("/")
    if len(parts) != 4 or not all(parts):
        sys.exit(f"Invalid target '{target}' — expected exactly "
                 "<namespace>/<model>/<type>/<script_name>")
    ns, model_name, type_, name = parts
    if type_ not in TYPES:
        sys.exit(f"Invalid <type> '{type_}' — must be one of: {' | '.join(TYPES)}")
    return ns, model_name, type_, name


# ---------------------------------------------------------------------------
# Step 3 — resolve the script.
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
# (<model>/<operation>/results.yaml) — never runnables. A runnable matches
# when its filename starts with "<script_name>." and has a script extension;
# *.config.json and data files (e.g. results.yaml) never do. Exactly one
# match must remain.
# ---------------------------------------------------------------------------
RUNNABLE_SUFFIXES = (".py", ".ipynb")


def resolve_script(ns, model_name, type_, name):
    """Return (service, script_path). service 'huggingface' == first-party."""
    model_dir = ROOT / ns / model_name
    if not model_dir.is_dir():
        sys.exit(f"ERROR: model folder not found: {ns}/{model_name}")

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
            service, ref = entry.get("service"), entry.get("reference")
            if not service or not ref:
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
            for p in runnables(ROOT / "external" / service / ref / type_):
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


# ---------------------------------------------------------------------------
# Step 4a — the default runner: Hugging Face Jobs, in-process.
# The hf CLI is a wrapper around the huggingface_hub library, so we call the
# library directly (HfApi.run_uv_job) instead of shelling out. Implements the
# same BaseService interface as every external service runner.
# ---------------------------------------------------------------------------
def build_hf_job_kwargs(options, script, token):
    """Translate a config.json 'options' block into run_uv_job(...) kwargs.

    Pure function (no huggingface_hub import) so it is unit-testable anywhere.
    Secrets: 'KEY=VALUE' passes the value; a bare 'KEY' forwards the logged-in
    token (mirroring the CLI's `--secrets HF_TOKEN` behaviour).
    """
    kwargs = {"script": str(script)}
    if options.get("script_args"):
        kwargs["script_args"] = list(options["script_args"])
    for key in ("flavor", "timeout", "image", "namespace"):
        if options.get(key):
            kwargs[key] = options[key]
    if options.get("python"):
        kwargs["python"] = options["python"]
    if options.get("env"):
        kwargs["env"] = dict(options["env"])
    if options.get("with"):
        kwargs["dependencies"] = list(options["with"])
    secrets = {}
    for entry in options.get("secrets", []):
        if "=" in entry:
            key, value = entry.split("=", 1)
            secrets[key] = value
        else:
            secrets[entry] = token          # bare name -> forward the login token
    if secrets:
        kwargs["secrets"] = secrets
    return kwargs


class HuggingFaceService(BaseService):
    """The default runner: submits first-party scripts as Hugging Face Jobs.

    Implements the full BaseService contract via the huggingface_hub library:
      run()           -> run id: the Hugging Face Job id
      get_status(id)  -> inspect_job(job_id): the job's stage
      list_runs()     -> list_jobs() for the current user
      list_datasets() -> list_datasets(author=<current user>)
      login()         -> mandates HF_TOKEN in the environment (workspace
                         .env, loaded at startup) — the only login route
    """

    @staticmethod
    def _hub():
        """The huggingface_hub module (installed by the launcher's uv sync)."""
        try:
            import huggingface_hub
        except ImportError:
            sys.exit("The huggingface_hub library is required (the hf CLI is only a "
                     "wrapper around it). It is installed by the launcher's `uv sync` "
                     "— run via ./run.sh, or: uv sync")
        return huggingface_hub

    # -- authentication -----------------------------------------------------
    def login(self) -> None:
        """HF_TOKEN in the environment (.env) is the only login route.

        huggingface_hub reads HF_TOKEN natively on every API call — nothing
        to store here; a bad token fails at the first call.
        """
        if not os.environ.get("HF_TOKEN"):
            sys.exit("HuggingFaceService: HF_TOKEN not set — put it in .env "
                     "(template: .env.example)")

    def is_logged_in(self) -> bool:
        """True when a stored/env HF token exists."""
        return self._hub().get_token() is not None

    # -- service entry point (guarded by BaseService.run) ---------------------
    def _run(self, script, model, type_, name):
        config = self._load_config(script, name)
        # The top-level "options" block is the ONLY source of run options —
        # nothing is assumed here and nothing is read from other parts of the
        # config.
        options = config.get("options")
        if not options:
            sys.exit(f"{script.parent / (name + '.config.json')} has no "
                     "top-level 'options' block")
        hub = self._hub()
        api = hub.HfApi()
        runner = getattr(api, "run_uv_job", None)
        if runner is None:
            sys.exit("This huggingface_hub version has no Jobs API (run_uv_job) — "
                     "bump its floor in pyproject.toml and re-run uv sync")

        kwargs = build_hf_job_kwargs(options, script, hub.get_token())
        # Only pass kwargs this huggingface_hub version understands.
        supported = set(inspect.signature(runner).parameters)
        for key in [k for k in kwargs if k not in supported and k != "script"]:
            print(f"note: option '{key}' not supported by this huggingface_hub "
                  "version — dropped")
            kwargs.pop(key)

        print(f"Submitting Hugging Face Job: {script.relative_to(self.root)} "
              f"(flavor={kwargs.get('flavor', '(not set in config -> API default)')})")
        job = runner(**kwargs)
        job_id = getattr(job, "id", job)
        print(f"Job submitted: id={job_id}  url={getattr(job, 'url', 'n/a')}")
        if options.get("detach"):
            print("Detached. Monitor with `hf jobs ps` / `hf jobs logs <job-id>`.")
        else:
            # Best-effort log streaming; the job keeps running if this fails.
            try:
                for line in api.fetch_job_logs(job_id=job_id):
                    print(line)
            except Exception as exc:
                print(f"(log streaming unavailable: {exc} — job still running; "
                      f"use `hf jobs logs {job_id}`)")
        print(f"Hub target: https://huggingface.co/{model}")
        return str(job_id)                 # the run id: the HF Job id

    # -- status / listings (guarded by the BaseService wrappers) --------------
    def _get_status(self, run_id):
        """The job's stage (e.g. RUNNING, COMPLETED, ERROR) for a job id."""
        api = self._hub().HfApi()
        try:
            job = api.inspect_job(job_id=run_id)
        except Exception as exc:
            sys.exit(f"Cannot get status of job '{run_id}': {exc}")
        status = getattr(job, "status", None)
        return str(getattr(status, "stage", status))

    def _list_runs(self):
        """The current user's Hugging Face Jobs, newest first."""
        api = self._hub().HfApi()
        runs = []
        for job in api.list_jobs():
            status = getattr(job, "status", None)
            command = getattr(job, "command", None) or []
            runs.append({"id": str(getattr(job, "id", "")),
                         "title": " ".join(map(str, command))[:80],
                         "status": str(getattr(status, "stage", status))})
        return runs

    def _list_datasets(self):
        """The current user's datasets on the Hub, as "<user>/<name>" ids."""
        api = self._hub().HfApi()
        user = api.whoami().get("name")
        return [str(d.id) for d in api.list_datasets(author=user)]


# ---------------------------------------------------------------------------
# Step 4b — service loading + dispatch.
# 'huggingface' maps to the built-in class above. Any other service is loaded
# dynamically from external/<service>/service.py, which must define exactly
# one class extending BaseService (class only — no main, no argparse). The
# instance's run(...) is called in-process; there is no subprocess.
# ---------------------------------------------------------------------------
def load_service(service):
    """Return a BaseService instance for the named service."""
    if service == "huggingface":
        return HuggingFaceService(ROOT)

    svc_path = ROOT / "external" / service / "service.py"
    if not svc_path.is_file():
        sys.exit(f"Service runner missing: external/{service}/service.py "
                 "(each external service needs exactly one)")
    spec = importlib.util.spec_from_file_location(f"external.{service}.service", svc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [obj for obj in vars(module).values()
               if isinstance(obj, type) and issubclass(obj, BaseService)
               and obj is not BaseService]
    if not classes:
        sys.exit(f"external/{service}/service.py defines no BaseService subclass "
                 "— every service runner must extend baseservice.BaseService")
    if len(classes) > 1:
        names = ", ".join(c.__name__ for c in classes)
        sys.exit(f"external/{service}/service.py defines multiple BaseService "
                 f"subclasses ({names}) — keep exactly one")
    return classes[0](ROOT)


def main():
    ns, model_name, type_, name = ask_target(sys.argv)
    service, script = resolve_script(ns, model_name, type_, name)
    run_id = load_service(service).run(script, f"{ns}/{model_name}", type_, name)
    print(f"run id: {run_id}   (use it with the service's get_status)")


if __name__ == "__main__":
    main()
