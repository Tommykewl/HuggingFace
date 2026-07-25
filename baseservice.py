"""baseservice.py — the interface every service runner implements.

A *service* is anything that can execute one of the workspace's pipeline
scripts and answer questions about its account: Hugging Face Jobs (built into
run.py), Kaggle (external/kaggle/service.py), and so on. Each external
service ships exactly one `external/<service>/service.py` that defines a
single class extending BaseService — nothing else: no main(), no argparse, no
side effects at import time. run.py imports the module dynamically, finds the
BaseService subclass, instantiates it with the workspace root, and calls its
methods in-process (no subprocess).

Public API (all concrete here, so the login guard cannot be bypassed —
each one calls require_login() and then delegates to the service's
protected _implementation):

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

Authentication:

  * login(token=None)
        The ONLY method that runs without being logged in. With a token,
        authenticates non-interactively; without one, starts the service's
        interactive flow.
  * is_logged_in() -> bool
        Cheap credential check used by the guard (and callable directly).

Shared helpers:

  * load_config(script, name)
        The runnable's adjacent <name>.config.json — the SINGLE source of run
        options (services must not assume values or read options elsewhere).
  * require_login()
        sys.exit()s with a service-specific hint when not logged in.
"""

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class BaseService(ABC):
    """Interface + shared helpers for service runners."""

    def __init__(self, root):
        self.root = Path(root)

    # -- public API (concrete: guard first, then delegate) -------------------
    def run(self, script: Path, model: str, type_: str, name: str) -> str:
        """Execute the resolved runnable; returns a unique run id."""
        self.require_login()
        return self._run(script, model, type_, name)

    def get_status(self, run_id: str) -> str:
        """Status of the run identified by run_id (as returned by run())."""
        self.require_login()
        return self._get_status(run_id)

    def list_runs(self) -> list:
        """The current user's runs: [{"id", "title", "status"}, ...]."""
        self.require_login()
        return self._list_runs()

    def list_datasets(self) -> list:
        """The current user's datasets on the service, as reference strings."""
        self.require_login()
        return self._list_datasets()

    # -- authentication ------------------------------------------------------
    @abstractmethod
    def login(self, token=None) -> None:
        """Authenticate: with `token` non-interactively, else interactively.

        The only operation allowed while logged out.
        """

    @abstractmethod
    def is_logged_in(self) -> bool:
        """True when usable credentials are present."""

    def require_login(self) -> None:
        """Exit with a clear hint unless the user is logged in."""
        if not self.is_logged_in():
            sys.exit(f"{type(self).__name__}: not logged in — call login() "
                     "first (pass a token, or no argument for the interactive flow)")

    # -- service implementations (guarded by the public wrappers above) ------
    @abstractmethod
    def _run(self, script: Path, model: str, type_: str, name: str) -> str: ...

    @abstractmethod
    def _get_status(self, run_id: str) -> str: ...

    @abstractmethod
    def _list_runs(self) -> list: ...

    @abstractmethod
    def _list_datasets(self) -> list: ...

    # -- shared helpers ------------------------------------------------------
    def load_config(self, script: Path, name: str) -> dict:
        """The runnable's adjacent <name>.config.json (single options source)."""
        config_path = script.parent / f"{name}.config.json"
        if not config_path.is_file():
            sys.exit(f"Missing config: {config_path} "
                     "(every runnable needs a <name>.config.json)")
        return json.load(open(config_path))
