"""Kaggle service runner — a single BaseService class (see workspace
baseservice.py); run.py imports this module dynamically and calls its methods
in-process. No main(), no argparse, no CLI subprocesses: the kaggle Python
library is used directly (the kaggle CLI is only a wrapper around it).

Implements the full BaseService contract:
  run()           -> run id "<user>/<kernel-slug>" (the pushed kernel ref)
  get_status(id)  -> kernels_status: QUEUED / RUNNING / COMPLETE / ERROR ...
  list_runs()     -> kernels_list(mine=True)
  list_datasets() -> dataset_list(mine=True)
  login(token)    -> stores the access token in ~/.kaggle/access_token
                     (token-based; interactive mode prompts for one — Kaggle
                     has no in-library browser flow); the other methods are
                     guarded by BaseService.require_login().

Given a resolved notebook (e.g. external/kaggle/<ref>/trainingandevaluation/
train.kaggle.ipynb), run():

  1. authenticates via the kaggle library (~/.kaggle/kaggle.json or
     ~/.kaggle/access_token) and reads the username;
  2. loads the notebook's adjacent <name>.config.json (kernel options:
     language, kernel_type, is_private, enable_gpu, machine_shape,
     enable_internet, dataset_sources, kernel_sources, optional pinned slug,
     top-level timeout_seconds);
  3. builds kernel-metadata.json purely from the config — bare dataset/kernel
     slugs get the username prefixed, and the kernel slug is: config "slug" >
     model name (for train) > <model>-<name>. Nothing is injected that the
     config does not declare (a warning is printed if no external-secrets
     dataset is listed, since the notebook then cannot read HF_TOKEN);
  4. verifies every mount BEFORE pushing (each dataset must exist, each source
     kernel must be COMPLETE — its output is the staged data), so a missing
     mount can never burn GPU time;
  5. pushes the kernel, which queues the run, and prints how to monitor it.

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
    def login(self, token=None) -> None:
        """Store a Kaggle access token (KGAT_...) in ~/.kaggle/access_token.

        With `token`, non-interactive; without, prompts for one (create it at
        https://www.kaggle.com/settings — the kaggle library has no in-library
        browser flow). kaggle.json credentials, if already present, also work.
        """
        if not token:
            token = input("Paste your Kaggle access token "
                          "(https://www.kaggle.com/settings): ").strip()
        if not token:
            sys.exit("No token provided — login aborted")
        cred_dir = Path.home() / ".kaggle"
        cred_dir.mkdir(mode=0o700, exist_ok=True)
        cred_file = cred_dir / "access_token"
        cred_file.write_text(token.strip() + "\n")
        cred_file.chmod(0o600)
        type(self)._api_cache = None       # force re-auth with the new token
        if not self.is_logged_in():
            sys.exit("Login failed — the token was stored but authentication "
                     "still fails; check the token and try again")
        print(f"Logged in to Kaggle as {self._api()[1]} (token in {cred_file})")

    def is_logged_in(self) -> bool:
        """True when the kaggle library can authenticate with stored creds."""
        try:
            return bool(self._api()[1])
        except SystemExit:
            return False

    # -- service entry point (guarded by BaseService.run) ---------------------
    def _run(self, script: Path, model: str, type_: str, name: str) -> str:
        config = self.load_config(script, name)
        model_name = model.split("/", 1)[1]

        api, user = self._api()
        meta = self._build_metadata(config, user, model_name, name)
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
    def _build_metadata(config: dict, user: str, model_name: str, name: str) -> dict:
        """Turn a notebook's config.json into Kaggle kernel-metadata.json."""
        km = config.get("kernel_metadata", {})

        def qualify(refs):
            return [r if "/" in r else f"{user}/{r}" for r in refs]

        slug = config.get("slug") or (model_name if name == "train" else f"{model_name}-{name}")
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
                         f"Its output is the staged data — run its prepare-data first, e.g.: "
                         f"./run.sh {model}/data-preparation/prepare-data")

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
