"""Kaggle service runner — a single BaseService class (see workspace
baseservice.py); run.py imports this module dynamically and calls its methods
in-process. No main(), no argparse, no CLI subprocesses: the kaggle Python
library is used directly (the kaggle CLI is only a wrapper around it).

Implements the full BaseService contract:
  run()           -> run id "<user>/<kernel-slug>" (the pushed kernel ref)
  get_status(id)  -> kernels_status: QUEUED / RUNNING / COMPLETE / ERROR ...
  list_runs()     -> kernels_list(mine=True)
  list_datasets() -> dataset_list(mine=True)
  login()         -> mandates Kaggle credentials in the environment
                     (workspace .env, loaded at startup) — the only login
                     route. Either credential kind works: KAGGLE_TOKEN (an
                     access token, stored to ~/.kaggle/access_token for the
                     library) or KAGGLE_USERNAME/KAGGLE_KEY (read natively
                     by the library).

The kaggle library (not kagglehub) is used deliberately: kagglehub has no
kernel push/status/list and no dataset-existence check — only auth,
dataset/model download+upload, and notebook-output download.

Given a resolved notebook (e.g. external/kaggle/<ref>/training/
index.kaggle.ipynb), run():

  1. authenticates via the kaggle library (~/.kaggle/kaggle.json or
     ~/.kaggle/access_token) and reads the username;
  2. _setup(): ensures the private external-secrets dataset exists on the
     account — if missing, it is created from the HF_TOKEN environment
     variable (loaded from .env by run.py at startup) and uploaded as a
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
  6. pushes the kernel, which queues the run, and prints how to monitor it.

Why the checks are strict: runs here cost real GPU quota. A push that would
start without its data mount would silently fall back to a many-small-files
Hub download that once consumed an entire 18-hour session.

Background on the pinned options (see also the config.json notes):
  * machine_shape must be NvidiaTeslaT4 for GPU runs — the API default GPU is
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
from pathlib import Path

from baseservice import BaseService

SECRETS_SLUG = "external-secrets"


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
If a training run finished but nothing was pushed to the Hub, the secrets
dataset wasn't readable — the model was saved to the kernel Output tab instead.""")
        return kernel                      # the run id: "<user>/<kernel-slug>"

    # -- status / listings (guarded by the BaseService wrappers) --------------
    def _get_status(self, run_id: str) -> str:
        """Status string for a kernel run id ("<user>/<kernel-slug>")."""
        api, _ = self._api()
        try:
            response = api.kernels_status(run_id)
        except Exception as exc:
            sys.exit(f"Cannot get status of '{run_id}': {exc}")
        status = str(getattr(response, "status", response))
        failure = getattr(response, "failure_message", None)
        return f"{status} ({failure})" if failure else status

    def _list_runs(self) -> list:
        """The current user's kernels, newest first."""
        api, _ = self._api()
        kernels = api.kernels_list(mine=True, page_size=50, sort_by="dateRun") or []
        return [{"id": str(k.ref), "title": str(getattr(k, "title", "") or ""),
                 "status": None}           # kernels_list carries no per-run status;
                for k in kernels if k]     # use get_status(id) for a specific run

    def _list_datasets(self) -> list:
        """The current user's datasets, as "<user>/<slug>" references."""
        api, _ = self._api()
        datasets = api.dataset_list(mine=True) or []
        return [str(d.ref) for d in datasets if d]

    # -- helpers ------------------------------------------------------------
    def _setup(self) -> None:
        """Ensure the private external-secrets dataset exists on the account.

        Training kernels read HF_TOKEN from it (Kaggle Secrets are dropped on
        every push; dataset mounts persist). When it is missing, it is created
        here from the HF_TOKEN environment variable — loaded from .env by
        run.py at startup — as a `secrets` file of KEY=VALUE lines, uploaded
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
                     "launcher's `uv sync` — run via ./run.sh, or: uv sync")
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
                         f"./run.sh {model}/<type>/prepare-data")

    @staticmethod
    def _push(api, notebook: Path, meta: dict, timeout_seconds=None) -> str:
        """Stage notebook + metadata in a temp dir and push (queues the run)."""
        with tempfile.TemporaryDirectory() as stage:
            shutil.copy(notebook, stage)
            meta["code_file"] = notebook.name
            with open(Path(stage) / "kernel-metadata.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            if timeout_seconds:            # config "timeout_seconds": run cap
                response = api.kernels_push(stage, timeout=str(timeout_seconds))
            else:
                response = api.kernels_push(stage)
        error = getattr(response, "error", None)
        if error:
            sys.exit(f"Kaggle push failed: {error}")
        version = getattr(response, "version_number", None)
        print(f"Kernel version {version or '?'} pushed.")
        return meta["id"]
