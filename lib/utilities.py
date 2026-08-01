"""Shared utilities for the workspace operations, in three sections:

  Help system      — helptexts live IN the operation classes (each
                     BaseOperation carries a _helptext list of (usage,
                     detail) entries); this section only AGGREGATES them:
                     the overview joins every usage line, `help <operation>
                     <entity>` prints the matching entries in full. `help`
                     itself is not an operation — lib/main.py calls
                     show_help directly.
  Service loading  — every service is a class implementing the BaseService
                     interface (lib/baseservice.py), shipped as exactly one
                     service.py (hf/service.py for huggingface,
                     <service>/service.py at the repo root for externals),
                     imported dynamically and called in-process. Credentials
                     are not checked upfront: every public service method
                     calls its service's login() first.
  Git plumbing     — THE single executor for git commands plus the
                     registered-submodule query built on it. Everything
                     git-related goes through git() — no other module
                     starts a git subprocess.
"""

import importlib.util
import os
import subprocess
import sys

from lib.baseservice import BaseService
from lib.config import ROOT


# ---------------------------------------------------------------------------
# Help system.
# ---------------------------------------------------------------------------
HELP_ENTRY = ("help [operation [entity]]",
              "Print the usage overview, or one operation's detailed usage.\n"
              "Example: help execute jobs | help status jobs | "
              "help list namespaces")


def all_entries(operations):
    """Every (usage, detail) entry: each operation's, then help's own."""
    entries = []
    for op in operations:
        entries.extend(op.help_entries())
    entries.append(HELP_ENTRY)
    return entries


def overview(operations):
    """The one-line usage of every operation, indented for printing."""
    return "\n  ".join(usage for usage, _ in all_entries(operations))


def usage_exit(problem, operations):
    sys.exit(f"{problem}\nOperations (details: help <operation> <entity>):"
             f"\n  {overview(operations)}")


def show_help(topic_words, operations):
    """The usage overview, or the matching operations' detailed usage.

    `help` alone prints every usage line; `help <operation> [entity]` prints
    every entry whose usage starts with the topic (so `help load` shows all
    three load entries, `help load models` exactly one). Also what a bare
    launcher invocation shows — help must work on a fresh clone with no .env.
    """
    if not topic_words:
        print("Usage: ./mlops.sh <operation> <entity> [args]   "
              "(Windows: .\\mlops.ps1)\n\n"
              f"Operations:\n  {overview(operations)}\n\n"
              "Details on one operation: help <operation> <entity> — "
              "e.g. help status jobs")
        return
    topic = " ".join(topic_words)
    matches = [f"Usage: {usage}\n\n{detail}"
               for usage, detail in all_entries(operations)
               if usage == topic or usage.startswith(topic + " ")]
    if not matches:
        usage_exit(f"help: unknown operation {topic!r}", operations)
    print("\n\n".join(matches))


# ---------------------------------------------------------------------------
# Service loading.
# ---------------------------------------------------------------------------
def load_service(service):
    """Return a BaseService instance for the named service.

    Every service — huggingface included — is loaded dynamically from its
    service.py: hf/service.py for huggingface, <service>/service.py at the
    repo root otherwise (e.g. kaggle/service.py).
    """
    if service == "huggingface":
        svc_path = ROOT / "hf" / "service.py"
    else:
        svc_path = ROOT / service / "service.py"
    if not svc_path.is_file():
        sys.exit(f"Service runner missing: {svc_path.relative_to(ROOT)} "
                 "(each service needs exactly one)")
    spec = importlib.util.spec_from_file_location(f"services.{service}", svc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [obj for obj in vars(module).values()
               if isinstance(obj, type) and issubclass(obj, BaseService)
               and obj is not BaseService]
    if not classes:
        sys.exit(f"{svc_path.relative_to(ROOT)} defines no BaseService subclass "
                 "— every service runner must extend lib.baseservice.BaseService")
    if len(classes) > 1:
        names = ", ".join(c.__name__ for c in classes)
        sys.exit(f"{svc_path.relative_to(ROOT)} defines multiple BaseService "
                 f"subclasses ({names}) — keep exactly one")
    return classes[0](ROOT)


def reachable_services():
    """Yield (name, instance) for every registered service — huggingface
    (hf/service.py) first, then each external that ships a service.py —
    that has usable credentials; the rest are noted and skipped, so one
    missing login never fails a whole sweep."""
    names = ["huggingface"]
    names += sorted(d.name for d in ROOT.iterdir()
                    if d.is_dir() and d.name != "hf"
                    and (d / "service.py").is_file())
    for name in names:
        service = load_service(name)
        if service.is_logged_in():
            yield name, service
        else:
            print(f"[{name}] skipped — no credentials in .env for this service")


# ---------------------------------------------------------------------------
# Git plumbing.
# ---------------------------------------------------------------------------
def git(*args, lfs_skip=False, capture=False, check=True):
    """THE single executor for git commands (always at the workspace root —
    target a submodule with -C). Returns the CompletedProcess.

    lfs_skip sets GIT_LFS_SKIP_SMUDGE=1 so submodule clones keep LFS files
    as pointers (weights/data are never materialized locally); capture
    collects stdout/stderr as text instead of streaming to the terminal;
    check=False hands the exit code back to the caller instead of exiting
    on failure.
    """
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1") if lfs_skip else None
    result = subprocess.run(["git", *args], cwd=ROOT, env=env,
                            capture_output=capture, text=capture)
    if check and result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed (exit {result.returncode})")
    return result


def registered_submodules():
    """Every submodule path registered in .gitmodules, as workspace-relative
    strings (found by path value — section names may lag behind moves).
    check=False: a missing/empty .gitmodules is a plain 'no submodules'."""
    result = git("config", "-f", ".gitmodules", "--get-regexp",
                 r"submodule\..*\.path", capture=True, check=False)
    return [line.split(maxsplit=1)[1]
            for line in result.stdout.splitlines() if " " in line]
