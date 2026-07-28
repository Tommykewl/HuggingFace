"""baseservice.py — the interface every service runner implements.

A *service* is anything that can execute one of the workspace's pipeline
scripts and answer questions about its account: Hugging Face Jobs (built into
run.py), Kaggle (external/kaggle/service.py), and so on. Each external
service ships exactly one `external/<service>/service.py` that defines a
single class extending BaseService — nothing else: no main(), no argparse, no
side effects at import time. run.py imports the module dynamically, finds the
BaseService subclass, instantiates it with the workspace root, and calls its
methods in-process (no subprocess).

Public API (all concrete here, so the login flow cannot be bypassed — each
one logs in first and then delegates to the service's protected
_implementation):

  * run(script, model, type_, name) -> run_id
        Execute the resolved runnable and return a unique run id usable with
        get_status() later (Kaggle: "<user>/<kernel-slug>"; Hugging Face:
        the job id).
  * get_status(run_id) -> str
        Current status of a run previously started (or any run the account
        can see) identified by that run id.
  * list_runs() -> list[dict]
        The current user's runs, newest first, as {"id", "title", "status"}
        dicts (fields best-effort; "id" is always usable with get_status).
  * list_datasets() -> list[str]
        The current user's datasets on the service, as reference strings.

Authentication — every public method calls login() first. login() is
service-specific and therefore abstract here (each SDK names and consumes
credentials differently), but every implementation follows one rule:
credentials are MANDATED as environment variables from the workspace .env
(template: .env.example, loaded by run.py at startup). Return only when the
environment holds usable credentials; sys.exit naming the missing
variable(s) otherwise. No interactive prompts, no other login routes.

Services implement:

  * login() — as above.
  * is_logged_in() -> bool — cheap credential check.

Shared helper (service-level, protected — a service's _run calls it only if
that service needs a config; a future config-less service simply doesn't):

  * _load_config(script, name)
        The runnable's adjacent <name>.config.json or <name>.config.yaml —
        the SINGLE source of run options and operation inputs (services must
        not assume values or read options elsewhere).
"""

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class BaseService(ABC):
    """Interface + shared helpers for service runners."""

    def __init__(self, root):
        self.root = Path(root)

    # -- public API (concrete: log in first, then delegate) ------------------
    def run(self, script: Path, model: str, type_: str, name: str) -> str:
        """Execute the resolved runnable; returns a unique run id."""
        self.login()
        return self._run(script, model, type_, name)

    def get_status(self, run_id: str) -> str:
        """Status of the run identified by run_id (as returned by run())."""
        self.login()
        return self._get_status(run_id)

    def list_runs(self) -> list:
        """The current user's runs: [{"id", "title", "status"}, ...]."""
        self.login()
        return self._list_runs()

    def list_datasets(self) -> list:
        """The current user's datasets on the service, as reference strings."""
        self.login()
        return self._list_datasets()

    # -- authentication ------------------------------------------------------
    @abstractmethod
    def login(self) -> None:
        """Authenticate from environment variables — the only login route.

        Called by every public method before doing anything else. An
        implementation must mandate its credentials as env vars (the
        workspace .env is loaded by run.py at startup), return only when
        they authenticate, and sys.exit naming the missing variable(s)
        otherwise. No interactive prompts or other fallbacks.
        """

    @abstractmethod
    def is_logged_in(self) -> bool:
        """True when usable credentials are present."""

    # -- service implementations (guarded by the public wrappers above) ------
    @abstractmethod
    def _run(self, script: Path, model: str, type_: str, name: str) -> str: ...

    @abstractmethod
    def _get_status(self, run_id: str) -> str: ...

    @abstractmethod
    def _list_runs(self) -> list: ...

    @abstractmethod
    def _list_datasets(self) -> list: ...

    # -- shared helpers (protected: for service implementations only) --------
    def _load_config(self, script: Path, name: str) -> dict:
        """The runnable's adjacent <name>.config.json OR <name>.config.yaml.

        The single source of run options and operation inputs. Protected on
        purpose: only a service's _run knows whether it needs a config, so
        calling (and thus requiring) one is a service-level decision — a
        config-less service simply never calls this.
        """
        for suffix, loader in ((".config.json", json.load),
                               (".config.yaml", None), (".config.yml", None)):
            config_path = script.parent / f"{name}{suffix}"
            if config_path.is_file():
                if loader is json.load:
                    return json.load(open(config_path))
                import yaml                # provided transitively; lazy on purpose
                return yaml.safe_load(open(config_path)) or {}
        sys.exit(f"Missing config: {script.parent / (name + '.config.json')} "
                 f"(or .config.yaml) — this service requires one per runnable")
