"""Kaggle service runner — a single BaseService class (see workspace
lib/baseservice.py); lib/main.py imports this module dynamically and calls its methods
in-process. No main(), no argparse, no CLI subprocesses: the kaggle Python
library is used directly (the kaggle CLI is only a wrapper around it).

Implements the full BaseService contract:
  run()           -> run id "<user>/<kernel-slug>" (the pushed kernel ref)
  get_status(id)  -> kernels_status: QUEUED / RUNNING / COMPLETE / ERROR ...
  list(entity)    -> "jobs": the job staging datasets (JOBS_MARKER
                     subtitle, filtered server-side); "datasets":
                     dataset_list(mine=True) minus the jobs datasets;
                     "namespaces": the logged-in username; nothing else
  create(entity, namespace, name) -> "jobs" only: the job's private
                     staging dataset (JOBS_MARKER subtitle — the source
                     of truth for the job's artifacts); everything else
                     errors (kernels appear via kernels_push, data
                     datasets/models via uploads)
  delete(entity, namespace, name) -> "jobs": delete the staging dataset
                     (marker-verified first); "datasets": dataset_delete;
                     "models": model_delete; nothing else
  load/unload(entity, namespace, name) -> not git-based: load downloads
                     into the gitignored kaggle/<ns>/<entity>/<name>
                     (datasets directly, jobs from their marker-verified
                     staging dataset); unload deletes that folder
  login()         -> mandates Kaggle credentials in the environment
                     (workspace .env, loaded at startup) — the only login
                     route. Either credential kind works: KAGGLE_TOKEN (an
                     access token, stored to ~/.kaggle/access_token for the
                     library) or KAGGLE_USERNAME/KAGGLE_KEY (read natively
                     by the library).

The kaggle library (not kagglehub) is used deliberately: kagglehub has no
kernel push/status/list and no dataset-existence check — only auth,
dataset/model download+upload, and notebook-output download.

Given a resolved notebook (e.g. kaggle/<username>/jobs/<ref>/training/
index.kaggle.ipynb), run():

  1. authenticates via the kaggle library (~/.kaggle/kaggle.json or
     ~/.kaggle/access_token) and reads the username;
  2. _setup(): ensures the private external-secrets dataset exists on the
     account — if missing, it is created from the HF_TOKEN environment
     variable (loaded from .env by lib/main.py at startup) and uploaded as a
     `secrets` file of KEY=VALUE lines;
  3. loads the notebook's adjacent <name>.config.json (kernel options:
     language, kernel_type, is_private, enable_gpu, machine_shape,
     enable_internet, dataset_sources, kernel_sources, optional pinned slug,
     top-level timeout_seconds);
  4. builds kernel-metadata.json purely from the config — bare dataset/kernel
     slugs get the username prefixed, and the kernel slug follows the naming
     convention: config "slug" override > derived "<type>-<name>" (e.g.
     training-index, evaluation-prepare-data), max 50 chars
     (Kaggle rejects longer with a bare 400). On Kaggle the kernels are
     grouped in a collection named after the model — maintained manually in
     the UI, since the kaggle API has no collections endpoint. Nothing is
     injected that the config does not declare (a warning is printed if no
     external-secrets dataset is listed, since the notebook then cannot read
     HF_TOKEN);
  5. verifies every mount BEFORE pushing (each dataset must exist, each source
     kernel must be COMPLETE — its output is the staged data), so a missing
     mount can never burn GPU time;
  6. pushes the kernel, which queues the job, and prints how to monitor it.

Why the checks are strict: jobs here cost real GPU quota. A push that would
start without its data mount would silently fall back to a many-small-files
Hub download that once consumed an entire 18-hour session.

Background on the pinned options (see also the config.json notes):
  * machine_shape must be NvidiaTeslaT4 for GPU jobs — the API default GPU is
    a P100 (sm_60), for which Kaggle's torch ships no CUDA kernels ("no kernel
    image is available for execution on the device");
  * the HF token rides in the private external-secrets dataset because Kaggle
    Secrets are dropped on every CLI push, while dataset references persist.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from lib.baseservice import BaseService
from lib.config import ROOT

SECRETS_SLUG = "external-secrets"
# A job's staging storage on Kaggle is a private dataset named after the
# job, marked by this string in its SUBTITLE — Kaggle rejects custom
# keywords/tags, and the subtitle is indexed by the server-side
# dataset_list(search=...) filter, so listing jobs is one API call.
JOBS_MARKER = "mlops-jobs"


class KaggleService(BaseService):
    """Runs a workspace notebook as a Kaggle kernel."""

    _api_cache = None                      # (api, user) after first successful auth

    # -- authentication -----------------------------------------------------
    def login(self) -> None:
        """Kaggle credentials in the environment (.env) are the only login
        route: KAGGLE_TOKEN (an access token — stored to
        ~/.kaggle/access_token, where the library reads it) or
        KAGGLE_USERNAME/KAGGLE_KEY (read natively by the library). A bad
        credential fails in _api() with the library's error.
        """
        token = os.environ.get("KAGGLE_TOKEN")
        if not token and not (os.environ.get("KAGGLE_USERNAME")
                              and os.environ.get("KAGGLE_KEY")):
            sys.exit("KaggleService: no Kaggle credentials — set KAGGLE_TOKEN "
                     "or KAGGLE_USERNAME/KAGGLE_KEY in .env "
                     "(template: .env.example)")
        if token and not self.is_logged_in():
            cred_file = Path.home() / ".kaggle" / "access_token"
            cred_file.parent.mkdir(mode=0o700, exist_ok=True)
            cred_file.write_text(token.strip() + "\n")
            cred_file.chmod(0o600)
            type(self)._api_cache = None   # re-auth with the new token

    def is_logged_in(self) -> bool:
        """True when the kaggle library can authenticate with stored creds."""
        try:
            return bool(self._api()[1])
        except SystemExit:
            return False

    # -- service entry point (guarded by BaseService.run) ---------------------
    def _run(self, script: Path, model: str, type_: str, name: str) -> str:
        self._setup()
        config = self._load_config(script, name)
        model_name = model.split("/", 1)[1]

        api, user = self._api()
        meta = self._build_metadata(config, user, model_name, type_, name)
        print(f"kernel: {meta['id']}   gpu={meta['enable_gpu']}"
              + (f" ({meta['machine_shape']})" if meta.get("machine_shape") else ""))
        print(f"dataset_sources: {meta['dataset_sources']}")
        print(f"kernel_sources:  {meta['kernel_sources']}")

        self._verify_mounts(api, meta, model)
        print("Pushing to Kaggle (this queues the run)...")
        kernel = self._push(api, script, meta, config.get("timeout_seconds"))

        print(f"""
{script.name} launched. Monitor it:
  kaggle kernels status {kernel}          # QUEUED / RUNNING / COMPLETE / ERROR
  open https://www.kaggle.com/code/{kernel}    # live logs

Hub target: https://huggingface.co/{model}
If a training job finished but nothing was pushed to the Hub, the secrets
dataset wasn't readable — the model was saved to the kernel Output tab instead.""")
        return kernel                      # the job id: "<user>/<kernel-slug>"

    # -- status / listings (guarded by the BaseService wrappers) --------------
    def _get_status(self, job_id: str) -> str:
        """Status string for a kernel job id ("<user>/<kernel-slug>")."""
        api, _ = self._api()
        try:
            response = api.kernels_status(job_id)
        except Exception as exc:
            sys.exit(f"Cannot get status of '{job_id}': {exc}")
        status = str(getattr(response, "status", response))
        failure = getattr(response, "failure_message", None)
        return f"{status} ({failure})" if failure else status

    def _list(self, entity: str) -> list:
        """The single listing implementation (see the BaseService contract).

        "jobs" -> the current user's kernels, newest first; "datasets" ->
        the current user's datasets as "<user>/<slug>" references;
        "namespaces" -> the logged-in username (Kaggle has no orgs — the
        account is its only namespace)."""
        api, user = self._api()
        if entity == "namespaces":
            return [user]
        if entity == "jobs":
            # The job staging datasets, filtered server-side by the
            # JOBS_MARKER subtitle (see its comment). Run instances are
            # tracked via `execute` / `status jobs`.
            return [str(d.ref) for d in self._jobs_datasets(api)]
        if entity == "datasets":
            # Plain data datasets only — the jobs staging datasets are a
            # different entity (`list jobs`).
            jobs = {str(d.ref) for d in self._jobs_datasets(api)}
            datasets = api.dataset_list(mine=True) or []
            return [str(d.ref) for d in datasets if d and str(d.ref) not in jobs]
        return None      # entity unknown here (e.g. Kaggle has no Spaces)

    @staticmethod
    def _jobs_datasets(api):
        """The account's job staging datasets: server-side search on the
        JOBS_MARKER subtitle, re-checked client-side (search also matches
        titles/descriptions)."""
        hits = api.dataset_list(mine=True, search=JOBS_MARKER) or []
        return [d for d in hits
                if d and JOBS_MARKER in str(getattr(d, "subtitle", "") or "")]

    def _create(self, entity: str, namespace: str, name: str) -> str:
        """Create <namespace>/<name> on Kaggle (see the BaseService
        contract).

        "jobs" -> the job's staging storage: a private dataset named after
        the job, marked as a jobs dataset via JOBS_MARKER in its subtitle
        (created with a placeholder manifest — Kaggle datasets need at
        least one file; artifacts arrive as later versions). Everything
        else errors out: kernels appear via `execute jobs` (kernels_push),
        data datasets and models via their folder-upload flows."""
        if entity != "jobs":
            sys.exit(f"Cannot create '{entity}' on kaggle — kernels are "
                     "created by `execute jobs` (kernels_push), datasets/"
                     "models by their upload flows; only jobs (their staging "
                     "dataset) are created by name")
        api, _ = self._api()
        ref = f"{namespace}/{name}"
        with tempfile.TemporaryDirectory() as stage:
            with open(Path(stage) / "dataset-metadata.json", "w") as fh:
                json.dump({"title": name, "id": ref,
                           "subtitle": f"{JOBS_MARKER} staging artifacts",
                           "licenses": [{"name": "CC0-1.0"}]}, fh)
            with open(Path(stage) / "README.md", "w") as fh:
                fh.write(f"# {name}\n\nmlops job staging. Layout: README.md; "
                         "one folder per stage (training/, evaluation/, ...) "
                         "holding the runnables; .runs/<run_id>/ — GENERATED "
                         "run records (script snapshot + config + "
                         "output.log), written only by mlops.\n")
            try:
                response = api.dataset_create_new(str(stage), public=False,
                                                  quiet=True, dir_mode="zip")
            except Exception as exc:
                sys.exit(f"Cannot create job staging dataset '{ref}': {exc}")
        error = getattr(response, "error", "") or ""
        if error:
            sys.exit(f"Cannot create job staging dataset '{ref}': {error}")
        # The marker subtitle rides in the create metadata above; verify it
        # stuck (it is how jobs datasets are identified — without it
        # `list jobs` cannot see it). Brand-new datasets 403 on every read
        # until processing finishes, so retry the metadata fetch itself.
        subtitle = ""
        for _ in range(12):
            try:
                with tempfile.TemporaryDirectory() as check:
                    api.dataset_metadata(ref, check)
                    info = json.load(open(Path(check) / "dataset-metadata.json"))
                    subtitle = (info.get("info") or info).get("subtitle") or ""
                break
            except Exception:
                time.sleep(5)
        if JOBS_MARKER not in subtitle:
            sys.exit(f"Created '{ref}' but the {JOBS_MARKER} subtitle marker "
                     "did not stick — set the subtitle manually, or the "
                     "dataset stays invisible to `list jobs`")
        return f"{ref} (staging dataset)"

    def _delete(self, entity: str, namespace: str, name: str) -> str:
        """Delete <namespace>/<name> on Kaggle (see the BaseService
        contract).

        "jobs" -> delete the job's staging dataset (verified to carry the
        JOBS_MARKER subtitle first, so a plain data dataset can never be
        deleted through the jobs entity), "datasets" -> dataset_delete,
        "models" -> model_delete; no_confirm skips the library's
        interactive prompt (mlops never prompts). Spaces and namespaces
        are not Kaggle concepts."""
        api, _ = self._api()
        ref = f"{namespace}/{name}"
        try:
            if entity == "jobs":
                if ref not in {str(d.ref) for d in self._jobs_datasets(api)}:
                    sys.exit(f"ERROR: '{ref}' is not a job staging dataset "
                             f"(no '{JOBS_MARKER}' subtitle) — see: list jobs")
                self._refuse_active_runs(name, "delete")
                api.dataset_delete(namespace, name, no_confirm=True)
                return f"{ref} (staging dataset)"
            if entity == "datasets":
                api.dataset_delete(namespace, name, no_confirm=True)
            elif entity == "models":
                api.model_delete(ref, no_confirm=True)
            else:
                sys.exit(f"Cannot delete '{entity}' on kaggle — no such concept")
        except SystemExit:
            raise
        except Exception as exc:
            sys.exit(f"Cannot delete {entity} '{ref}' on kaggle: {exc}")
        return ref

    # -- local materialization (guarded by BaseService.load/unload) ----------
    def _local_dir(self, entity, namespace, name):
        """Where the entity lives locally: kaggle/<ns>/<entity>/<name>
        (jobs: the stage-layout folder execute resolves scripts from).
        All of kaggle/<ns>/ is gitignored — remote-backed caches only."""
        return ROOT / "kaggle" / namespace / entity / name

    def _load(self, entity: str, namespace: str, name: str) -> str:
        """Materialize <namespace>/<name> locally (see the BaseService
        contract).

        Kaggle repos are not git-based, so loading is a download into the
        gitignored kaggle/<ns>/<entity>/<name>: "datasets" via
        dataset_download_files; "jobs" from the job's staging dataset (the
        source of truth — must exist, `create jobs` makes it; verified via
        its JOBS_MARKER subtitle). "models" cannot be loaded by name alone
        (a download needs framework/instance/version)."""
        api, _ = self._api()
        ref = f"{namespace}/{name}"
        target = self._local_dir(entity, namespace, name)
        rel = target.relative_to(ROOT)
        if entity not in ("datasets", "jobs"):
            sys.exit(f"Cannot load '{entity}' on kaggle — datasets and jobs "
                     "only (models need framework/instance/version, which "
                     "load's <name> cannot express)")
        if target.is_dir() and any(target.iterdir()):
            sys.exit(f"'{rel}' is already loaded — nothing to do")
        if entity == "jobs" and ref not in {str(d.ref)
                                            for d in self._jobs_datasets(api)}:
            sys.exit(f"ERROR: no job staging dataset '{ref}' — the staging "
                     "storage is the source of truth for jobs and is "
                     "created by `create jobs`")
        target.mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_files(ref, path=str(target), unzip=True,
                                       quiet=True)
        except Exception as exc:
            shutil.rmtree(target, ignore_errors=True)
            sys.exit(f"Cannot download '{ref}' to {rel}: {exc}")
        return f"{rel}  <->  {ref}"

    def _unload(self, entity: str, namespace: str, name: str,
                force: bool = False) -> str:
        """Remove the local copy of <namespace>/<name> (see the BaseService
        contract): a plain folder delete — Kaggle keeps the content (for
        jobs, the staging dataset is the source of truth; refused while
        any of the job's kernels is queued/running). force is meaningless
        here (nothing is git-registered) and ignored."""
        if entity not in ("datasets", "jobs"):
            sys.exit(f"Cannot unload '{entity}' on kaggle — datasets and "
                     "jobs only")
        if entity == "jobs":
            self._refuse_active_runs(name, "unload")
        target = self._local_dir(entity, namespace, name)
        rel = target.relative_to(ROOT)
        if not target.is_dir():
            sys.exit(f"'{rel}' is not loaded — nothing to do")
        shutil.rmtree(target)
        return (f"{rel} (deleted locally — kaggle keeps the content; "
                f"`load {entity}` restores it)")

    # -- run recording (guarded by BaseService.record_run/sync_logs) ---------
    def _staging_version(self, api, namespace, name, add_files, notes):
        """Publish a new version of job <name>'s staging dataset: its
        CURRENT remote content plus add_files ({relpath: Path | bytes}).
        Sourced from a fresh staging download in a temp dir — NEVER from
        the local kaggle/<ns>/jobs cache: .runs is generated, and manual
        local edits must never reach the staging."""
        ref = f"{namespace}/{name}"
        with tempfile.TemporaryDirectory() as stage:
            api.dataset_download_files(ref, path=stage, unzip=True, quiet=True)
            for relpath, content in add_files.items():
                dest = Path(stage) / relpath
                dest.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    dest.write_bytes(content)
                else:
                    shutil.copy(content, dest)
            with open(Path(stage) / "dataset-metadata.json", "w") as fh:
                json.dump({"title": name, "id": ref,
                           "subtitle": f"{JOBS_MARKER} staging artifacts",
                           "licenses": [{"name": "CC0-1.0"}]}, fh)
            api.dataset_create_version(str(stage), version_notes=notes,
                                       quiet=True, dir_mode="zip")

    def _record_run(self, namespace, job_name, run_id, script):
        """Snapshot a submitted run into the staging's .runs/ (see the
        BaseService contract). The run folder is named by the kernel slug
        (kaggle run ids are '<user>/<slug>'; the user is constant), and
        holds the executed script + its adjacent config.json. Re-pushing
        the same kernel reuses the slug, so the snapshot tracks the latest
        push — matching Kaggle's one-kernel-many-versions model."""
        api, _ = self._api()
        if f"{namespace}/{job_name}" not in {str(d.ref)
                                             for d in self._jobs_datasets(api)}:
            return (f"run NOT recorded: no staging dataset for job "
                    f"'{namespace}/{job_name}' — create jobs first")
        slug = run_id.rpartition("/")[2]
        add = {f".runs/{slug}/{script.name}": script}
        config = script.parent / (script.name.split(".")[0] + ".config.json")
        if config.is_file():
            add[f".runs/{slug}/{config.name}"] = config
        self._staging_version(api, namespace, job_name, add,
                              f"record run {slug}")
        return f"run recorded: {namespace}/{job_name} .runs/{slug}/"

    def _sync_logs(self, run_id):
        """Write a terminal kernel run's log as output.log into its
        .runs/<slug>/ staging entry (see the BaseService contract).
        Returns None while the kernel is still queued/running or when its
        slug maps to no job staging dataset ('<job name>-...')."""
        api, user = self._api()
        ref = run_id if "/" in run_id else f"{user}/{run_id}"
        status = str(getattr(api.kernels_status(ref), "status", "")).lower()
        if status in ("queued", "running"):
            return None
        slug = ref.rpartition("/")[2]
        job_name = next(
            (str(d.ref).partition("/")[2] for d in self._jobs_datasets(api)
             if slug.startswith(str(d.ref).partition("/")[2] + "-")), None)
        if job_name is None:
            return None                    # run belongs to no staged job
        with tempfile.TemporaryDirectory() as out:
            try:
                # file_pattern (a REGEX) limits the download to the run log
                # — without it kernels_output pulls EVERY output artifact
                # (a training kernel's saved models can be huge).
                api.kernels_output(ref, path=out, file_pattern=r".*\.log$",
                                   quiet=True)
            except Exception as exc:
                # Non-fatal: the caller's status query succeeded — a log
                # fetch hiccup (rate limit, transient API error) just means
                # the .runs entry completes on a later `status jobs`.
                return f"output.log NOT synced for '{ref}': {exc}"
            logs = sorted(Path(out).glob("*.log"))
            content = logs[0].read_bytes() if logs else b""
        self._staging_version(api, user, job_name,
                              {f".runs/{slug}/output.log": content},
                              f"output.log for {slug}")
        return f"synced: {user}/{job_name} .runs/{slug}/output.log"

    def _refuse_active_runs(self, name, verb):
        """Block unload/delete of job <name> while any of its kernels is
        still queued or running. A job's kernels are slugged
        '<job name>-<type>-<script>' (see run()), so the name prefix
        scopes the sweep; kernels_list carries no status, so each
        candidate is checked via kernels_status."""
        api, user = self._api()
        active = []
        for k in api.kernels_list(mine=True, page_size=50, search=name) or []:
            ref = str(k.ref) if k else ""
            if not ref.startswith(f"{user}/{name}-"):
                continue
            try:
                status = str(getattr(api.kernels_status(ref), "status", ""))
            except Exception:
                continue               # unreadable status never blocks
            if status.lower() in ("queued", "running"):
                active.append(f"{ref} ({status})")
        if active:
            sys.exit(f"ERROR: cannot {verb} job '{name}' — kernels still "
                     "active:\n  " + "\n  ".join(active))

    # -- helpers ------------------------------------------------------------
    def _setup(self) -> None:
        """Ensure the private external-secrets dataset exists on the account.

        Training kernels read HF_TOKEN from it (Kaggle Secrets are dropped on
        every push; dataset mounts persist). When it is missing, it is created
        here from the HF_TOKEN environment variable — loaded from .env by
        lib/main.py at startup — as a `secrets` file of KEY=VALUE lines, uploaded
        private. Idempotent: a no-op when the dataset already exists.
        """
        api, user = self._api()
        ref = f"{user}/{SECRETS_SLUG}"
        try:
            api.dataset_status(ref)        # raises when the dataset doesn't exist
            return
        except Exception:
            pass
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            sys.exit(f"Dataset '{ref}' does not exist and HF_TOKEN is not set, "
                     "so it cannot be created — set HF_TOKEN in .env "
                     "(template: .env.example)")
        print(f"Dataset {ref} not found — creating it (private) from HF_TOKEN...")
        with tempfile.TemporaryDirectory() as stage:
            (Path(stage) / "secrets").write_text(f"HF_TOKEN={hf_token}\n")
            with open(Path(stage) / "dataset-metadata.json", "w") as fh:
                json.dump({"title": SECRETS_SLUG, "id": ref,
                           "licenses": [{"name": "CC0-1.0"}]}, fh)
            result = api.dataset_create_new(folder=stage, public=False)
        error = getattr(result, "error", None)
        if error:
            sys.exit(f"Could not create dataset {ref}: {error}")
        print(f"Created private dataset {ref} (a 'secrets' file of KEY=VALUE "
              "lines — add more keys on Kaggle directly as needed).")

    @classmethod
    def _api(cls):
        """An authenticated KaggleApi instance and the account's username."""
        if cls._api_cache is not None:
            return cls._api_cache
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            sys.exit("The kaggle Python library is required. It is installed by the "
                     "launcher's `uv sync` — run via ./mlops.sh, or: uv sync")
        api = KaggleApi()
        try:
            api.authenticate()
        except Exception as exc:
            sys.exit(f"Kaggle authentication failed ({exc}) — put kaggle.json or "
                     "access_token in ~/.kaggle/, or call login()")
        user = ((getattr(api, "config_values", {}) or {}).get("username")
                or api.get_config_value("username"))
        if not user:
            sys.exit("Kaggle authentication gave no username — check ~/.kaggle/ credentials")
        cls._api_cache = (api, user)
        return cls._api_cache

    @staticmethod
    def _build_metadata(config: dict, user: str, model_name: str, type_: str,
                        name: str) -> dict:
        """Turn a notebook's config.json into Kaggle kernel-metadata.json.

        Kernel naming convention: the stage's main runnable (index) becomes
        "<model>-<type>" (e.g. rtdetr-sportsmot-training); every other
        runnable becomes "<model>-<type>-<name>" (e.g.
        rtdetr-sportsmot-training-prepare-data). Overridable per config via
        "slug" — needed when the derived name would exceed Kaggle's
        50-character slug limit. Kernels are grouped in a Kaggle collection
        named after the model, maintained manually in the UI (the kaggle API
        has no collections endpoint).
        """
        km = config.get("kernel_metadata", {})

        def qualify(refs):
            return [r if "/" in r else f"{user}/{r}" for r in refs]

        slug = config.get("slug") or (
            f"{model_name}-{type_}" if name == "index"
            else f"{model_name}-{type_}-{name}")
        if len(slug) > 50:
            sys.exit(f"Kernel slug '{slug}' is {len(slug)} chars — Kaggle "
                     "rejects slugs over 50 with a bare 400. Pin a shorter "
                     "'slug' in the config.")
        datasets = qualify(km.get("dataset_sources", []))
        if not any(d.endswith(f"/{SECRETS_SLUG}") for d in datasets):
            # Not auto-added: the config is the single source of truth for mounts.
            print(f"WARNING: no {SECRETS_SLUG} dataset in dataset_sources — the "
                  "notebook will not find HF_TOKEN and cannot push to the Hub.")
        # Every metadata field comes from the config's kernel_metadata block; the
        # only derived values are id/title (from slug) and code_file (the notebook).
        meta = {
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": None,             # filled in by _push() after staging
            "language": km.get("language", "python"),
            "kernel_type": km.get("kernel_type", "notebook"),
            "is_private": bool(km.get("is_private", True)),
            "enable_gpu": bool(km.get("enable_gpu", False)),
            "enable_internet": bool(km.get("enable_internet", True)),
            "dataset_sources": datasets,
            "competition_sources": km.get("competition_sources", []),
            "kernel_sources": qualify(km.get("kernel_sources", [])),
            "model_sources": km.get("model_sources", []),
        }
        if km.get("machine_shape"):
            meta["machine_shape"] = km["machine_shape"]
        return meta

    @staticmethod
    def _verify_mounts(api, meta: dict, model: str) -> None:
        """Abort BEFORE pushing if any declared mount is missing or not ready."""
        for ref in meta["dataset_sources"]:
            try:
                api.dataset_status(ref)    # raises when the dataset doesn't exist
            except Exception:
                msg = f"Dataset '{ref}' (from the config) not found on Kaggle."
                if ref.endswith(SECRETS_SLUG):
                    msg += (f"\nCreate the private {SECRETS_SLUG} dataset first "
                            "(a 'secrets' file with HF_TOKEN=...).")
                sys.exit(msg)
        for ref in meta["kernel_sources"]:
            try:
                status = str(api.kernels_status(ref).status)
            except Exception as exc:
                sys.exit(f"Kernel '{ref}' (from the config) not found on Kaggle ({exc}).")
            if "COMPLETE" not in status:
                sys.exit(f"Kernel '{ref}' (from the config) is not COMPLETE "
                         f"(status: {status}).\n"
                         f"Its output is the staged data — run its prepare-data staging script first, e.g.: "
                         f"./mlops.sh execute jobs {model}/<type>/prepare-data")

    @staticmethod
    def _push(api, notebook: Path, meta: dict, timeout_seconds=None) -> str:
        """Stage notebook + metadata in a temp dir and push (queues the job)."""
        with tempfile.TemporaryDirectory() as stage:
            shutil.copy(notebook, stage)
            meta["code_file"] = notebook.name
            with open(Path(stage) / "kernel-metadata.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            if timeout_seconds:            # config "timeout_seconds": job cap
                response = api.kernels_push(stage, timeout=str(timeout_seconds))
            else:
                response = api.kernels_push(stage)
        error = getattr(response, "error", None)
        if error:
            sys.exit(f"Kaggle push failed: {error}")
        version = getattr(response, "version_number", None)
        print(f"Kernel version {version or '?'} pushed.")
        return meta["id"]
