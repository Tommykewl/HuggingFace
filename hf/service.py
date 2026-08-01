"""Hugging Face service runner — a single BaseService class (see workspace
lib/baseservice.py); lib/main.py imports this module dynamically — exactly like an
external service's <service>/service.py, except this one lives at
hf/service.py, next to the Hub submodules it operates on — and calls its
methods in-process. No main(), no argparse, no CLI subprocesses: the
huggingface_hub Python library is used directly (the hf CLI is only a
wrapper around it).

Implements the full BaseService contract:
  run()           -> run id: the Hugging Face Job id (submits first-party
                     scripts as Hugging Face Jobs via HfApi.run_uv_job)
  get_status(id)  -> inspect_job(job_id): the job's stage
  list(entity)    -> the single listing method; beyond the contract's
                     "jobs" (list_jobs) it supports the Hub-only entities
                     "models" / "spaces" / "datasets" ({namespace: [ids]}
                     per account namespace) and "namespaces" (the
                     account's user + orgs)
  create(entity, name) -> create_repo for the repo entities (Spaces get
                     the Gradio SDK); jobs error out (submitted via
                     `execute jobs`, never created by name)
  delete(entity, name) -> delete_repo for the repo entities; "jobs"
                     cancels the Hub Job (they cannot be erased)
  login()         -> mandates HF_TOKEN in the environment (workspace
                     .env, loaded by lib/main.py at startup) — the only login
                     route
"""

import inspect
import os
import sys

from lib.baseservice import BaseService
from lib.config import REPO_KINDS


def hub():
    """The huggingface_hub module (installed by the launcher's uv sync)."""
    try:
        import huggingface_hub
    except ImportError:
        sys.exit("The huggingface_hub library is required (the hf CLI is only a "
                 "wrapper around it). It is installed by the launcher's `uv sync` "
                 "— run via ./mlops.sh, or: uv sync")
    return huggingface_hub


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
    """The default runner: submits first-party scripts as Hugging Face Jobs."""

    _hub = staticmethod(hub)

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
        hub_ = self._hub()
        api = hub_.HfApi()
        runner = getattr(api, "run_uv_job", None)
        if runner is None:
            sys.exit("This huggingface_hub version has no Jobs API (run_uv_job) — "
                     "bump its floor in pyproject.toml and re-run uv sync")

        kwargs = build_hf_job_kwargs(options, script, hub_.get_token())
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
            print("Detached. Monitor with `status jobs <job-id>`.")
        else:
            # Best-effort log streaming; the job keeps running if this fails.
            try:
                for line in api.fetch_job_logs(job_id=job_id):
                    print(line)
            except Exception as exc:
                print(f"(log streaming unavailable: {exc} — job still running; "
                      f"use `status jobs {job_id}`)")
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

    def _list(self, entity: str):
        """The single listing implementation (see the BaseService contract).

        "jobs" -> the current user's Hugging Face Jobs, newest first;
        "namespaces" -> the account's namespaces (the user + every org);
        "models" / "spaces" / "datasets" -> {namespace: [repo ids]} for
        every namespace of the account — one branch for all three, since
        the HfApi calls differ only in name (list_models / list_spaces /
        list_datasets)."""
        api = self._hub().HfApi()
        if entity == "jobs":
            jobs = []
            for job in api.list_jobs():
                status = getattr(job, "status", None)
                command = getattr(job, "command", None) or []
                jobs.append({"id": str(getattr(job, "id", "")),
                             "title": " ".join(map(str, command))[:80],
                             "status": str(getattr(status, "stage", status))})
            return jobs
        if entity == "namespaces":
            who = api.whoami()
            names = [who.get("name")]
            names += [org.get("name") for org in who.get("orgs", [])]
            return [n for n in names if n]
        if entity in ("models", "spaces", "datasets"):
            list_repos = getattr(api, f"list_{entity}")
            return {ns: [str(r.id) for r in list_repos(author=ns)]
                    for ns in self._list("namespaces")}
        return None                        # entity unknown to this service

    def _create(self, entity: str, namespace: str, name: str) -> str:
        """Create <namespace>/<name> on the Hub (see the BaseService
        contract).

        "models" / "spaces" / "datasets" -> create_repo (Spaces get the
        Gradio SDK — the workspace default). "jobs" cannot be created by
        name — they are submitted from a runnable via `execute jobs`."""
        if entity in REPO_KINDS:
            repo_type = REPO_KINDS[entity][0]
            repo_id = f"{namespace}/{name}"
            kwargs = {"repo_id": repo_id, "repo_type": repo_type}
            if repo_type == "space":
                kwargs["space_sdk"] = "gradio"
            try:
                return str(self._hub().HfApi().create_repo(**kwargs))
            except Exception as exc:
                sys.exit(f"Cannot create {repo_type} '{repo_id}' on the Hub: {exc}")
        sys.exit(f"Cannot create '{entity}' on huggingface — jobs are "
                 "submitted from a runnable via `execute jobs`, never "
                 "created by name")

    def _delete(self, entity: str, namespace: str, name: str) -> str:
        """Delete <namespace>/<name> on the Hub (see the BaseService
        contract).

        "models" / "spaces" / "datasets" -> delete_repo. "jobs" ->
        cancel_job in <namespace> (<name> is the job id): a Hub Job cannot
        be erased, so cancelling it is its deletion."""
        api = self._hub().HfApi()
        if entity in REPO_KINDS:
            repo_type = REPO_KINDS[entity][0]
            repo_id = f"{namespace}/{name}"
            try:
                api.delete_repo(repo_id=repo_id, repo_type=repo_type)
            except Exception as exc:
                sys.exit(f"Cannot delete {repo_type} '{repo_id}' on the Hub: {exc}")
            return repo_id
        if entity == "jobs":
            try:
                api.cancel_job(job_id=name, namespace=namespace)
            except Exception as exc:
                sys.exit(f"Cannot cancel job '{name}' in '{namespace}': {exc}")
            return f"{name} (cancelled — Hub Jobs cannot be erased)"
        sys.exit(f"Cannot delete '{entity}' on huggingface — no such concept")
