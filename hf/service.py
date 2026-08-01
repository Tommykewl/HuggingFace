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
  list(entity)    -> the single listing method: "models" / "spaces" /
                     "datasets" ({namespace: [ids]} per account
                     namespace), "namespaces" (the account's user + orgs)
                     and "jobs" (the job staging buckets per namespace —
                     the staging IS the job entity)
  create/delete(entity, namespace, name) -> create_repo / delete_repo for
                     the repo entities (Spaces get the Gradio SDK); "jobs"
                     creates/deletes the private staging bucket
                     <namespace>/<name> (the source of truth for the
                     job's artifacts)
  load/unload(entity, namespace, name) -> repo entities materialize as
                     git submodules under hf/ (unload: deinit, or full
                     removal with force); "jobs" sync from / delete the
                     gitignored local cache hf/<ns>/jobs/<name>
  login()         -> mandates HF_TOKEN in the environment (workspace
                     .env, loaded by lib/main.py at startup) — the only login
                     route
"""

import inspect
import os
import shutil
import sys

from lib.baseservice import BaseService
from lib.config import REPO_KINDS, ROOT
from lib.utilities import git, registered_submodules


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
            # Jobs are staged in storage buckets (<namespace>/<job name>) —
            # the staging is the entity, so listing jobs lists the buckets.
            # (Run instances are tracked via `execute` / `status jobs`.)
            return {ns: [str(b.id) for b in api.list_buckets(namespace=ns)]
                    for ns in self._list("namespaces")}
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
        if entity == "jobs":
            # A job's staging storage IS the job entity: a private storage
            # bucket <namespace>/<name> holding the job's artifacts — the
            # source of truth `load jobs` pulls from.
            bucket_id = f"{namespace}/{name}"
            try:
                self._hub().HfApi().create_bucket(bucket_id, private=True)
            except Exception as exc:
                sys.exit(f"Cannot create job staging bucket '{bucket_id}': {exc}")
            return f"{bucket_id} (staging bucket)"
        sys.exit(f"Cannot create '{entity}' on huggingface — no such concept")

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
            # Deleting a job deletes its staging bucket — artifacts and all.
            bucket_id = f"{namespace}/{name}"
            try:
                api.delete_bucket(bucket_id)
            except Exception as exc:
                sys.exit(f"Cannot delete job staging bucket '{bucket_id}': {exc}")
            return f"{bucket_id} (staging bucket)"
        sys.exit(f"Cannot delete '{entity}' on huggingface — no such concept")

    # -- local materialization (guarded by BaseService.load/unload) ----------
    def _local_dir(self, entity, namespace, name):
        """Where the entity lives locally: hf/<namespace>/<entity>/<name>."""
        return ROOT / "hf" / namespace / entity / name

    def _check_namespace(self, namespace):
        """<namespace> must be one of the account's namespaces."""
        namespaces = self._list("namespaces")
        if namespace not in namespaces:
            sys.exit(f"ERROR: '{namespace}' is not a namespace of this "
                     f"account on huggingface ({', '.join(namespaces)}) — "
                     "see: list namespaces")

    def _load(self, entity: str, namespace: str, name: str) -> str:
        """Materialize <namespace>/<name> locally (see the BaseService
        contract).

        Repo entities (models/spaces/datasets) are git repos: `git
        submodule add`ed (or `update --init`ed when already registered) at
        hf/<ns>/<entity>/<name> with GIT_LFS_SKIP_SMUDGE=1 so LFS payloads
        stay pointers. "jobs" syncs the job's staging bucket — the source
        of truth, which must already exist (`create jobs`) — into
        hf/<ns>/jobs/<name> (gitignored: remote-backed cache)."""
        api = self._hub().HfApi()
        self._check_namespace(namespace)
        target = self._local_dir(entity, namespace, name)
        rel = f"hf/{namespace}/{entity}/{name}"
        if entity in REPO_KINDS:
            repo_type, url_prefix = REPO_KINDS[entity]
            if not api.repo_exists(f"{namespace}/{name}", repo_type=repo_type):
                sys.exit(f"ERROR: {repo_type} '{namespace}/{name}' does not "
                         f"exist on the Hub — see: list {entity}")
            if target.is_dir() and any(target.iterdir()):
                sys.exit(f"'{rel}' is already loaded — nothing to do")
            url = f"{url_prefix}{namespace}/{name}"
            if rel in registered_submodules():
                # Registered in .gitmodules — just init it.
                git("submodule", "update", "--init", rel, lfs_skip=True)
            else:
                git("submodule", "add", url, rel, lfs_skip=True)
                print("note: the new submodule is staged (.gitmodules + "
                      "gitlink) — commit to keep it tracked")
            return f"{rel}  <->  {url}"
        if entity == "jobs":
            bucket_id = f"{namespace}/{name}"
            try:
                api.bucket_info(bucket_id)
            except Exception as exc:
                sys.exit(f"ERROR: no staging bucket '{bucket_id}' — the "
                         "staging storage is the source of truth for jobs "
                         f"and is created by `create jobs` ({exc})")
            target.mkdir(parents=True, exist_ok=True)
            try:
                api.sync_bucket(source=f"hf://buckets/{bucket_id}",
                                dest=str(target))
            except Exception as exc:
                sys.exit(f"Cannot sync bucket '{bucket_id}' to {rel}: {exc}")
            return f"{rel}  <->  hf://buckets/{bucket_id}"
        sys.exit(f"Cannot load '{entity}' on huggingface — no such concept")

    def _unload(self, entity: str, namespace: str, name: str,
                force: bool = False) -> str:
        """Remove the local copy of <namespace>/<name> (see the BaseService
        contract).

        Repo entities: `git submodule deinit` (stays registered) — with
        force, a complete removal: gitlink, .gitmodules entry and the
        .git/modules cache (blocked on unpushed commits, since the Hub repo
        becomes the only copy). Both refuse on uncommitted changes. "jobs":
        the local folder is deleted — the staging bucket keeps the
        artifacts (it is the source of truth)."""
        target = self._local_dir(entity, namespace, name)
        rel = f"hf/{namespace}/{entity}/{name}"
        if entity in REPO_KINDS:
            if rel not in registered_submodules():
                sys.exit(f"ERROR: '{rel}' is not a tracked submodule — "
                         f"see: list {entity}")
            if not (target.is_dir() and any(target.iterdir())):
                sys.exit(f"'{rel}' is not loaded — nothing to do")
            # Never discard local work: deinit -f would drop uncommitted
            # changes.
            dirty = git("-C", str(target), "status", "--porcelain",
                        capture=True)
            if dirty.stdout.strip():
                sys.exit(f"ERROR: '{rel}' has uncommitted changes — commit "
                         "and push them to the Hub before unloading:\n"
                         + dirty.stdout.rstrip())
            if not force:
                git("submodule", "deinit", "-f", rel)
                return f"{rel} (still registered — `load` restores it)"
            # force: the Hub repo becomes the only remaining copy, so
            # unpushed commits block too.
            unpushed = git("-C", str(target), "log", "--oneline",
                           "--branches", "--not", "--remotes", capture=True)
            if unpushed.stdout.strip():
                sys.exit(f"ERROR: '{rel}' has commits not pushed to the Hub "
                         "— push them before unloading with -f:\n"
                         + unpushed.stdout.rstrip())
            # Resolve the .git/modules cache path BEFORE git rm drops the
            # .gitmodules entry the name comes from.
            modname = self._submodule_name(rel)
            git("submodule", "deinit", "-f", rel)
            # git rm removes the worktree, the gitlink, and the .gitmodules
            # entry, staging all of it.
            git("rm", "-f", rel)
            if modname:
                shutil.rmtree(ROOT / ".git" / "modules" / modname,
                              ignore_errors=True)
            return f"{rel} (removed — staged; commit to share the removal)"
        if entity == "jobs":
            if not target.is_dir():
                sys.exit(f"'{rel}' is not loaded — nothing to do")
            shutil.rmtree(target)
            return f"{rel} (deleted locally — the staging bucket keeps the artifacts)"
        sys.exit(f"Cannot unload '{entity}' on huggingface — no such concept")

    @staticmethod
    def _submodule_name(rel):
        """The .gitmodules section name whose path is rel (name and path
        can differ — `submodule add` derives the name, and moves may
        desync them)."""
        result = git("config", "-f", ".gitmodules", "--get-regexp",
                     r"submodule\..*\.path", capture=True, check=False)
        for line in result.stdout.splitlines():
            key, _, path = line.partition(" ")
            if path == rel:
                return key[len("submodule."):-len(".path")]
        return None
