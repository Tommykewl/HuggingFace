#!/usr/bin/env python3
"""lib/main.py — the workspace operations dispatcher (the entrypoint (steps 2–3). Invoked by
the mlops.sh / mlops.ps1 launchers after they complete step 1 (prerequisite
checks); can also be called directly:

    python3 lib/main.py <operation> [args]

Operations are verbs; the ENTITY they act on (namespaces, models, spaces,
datasets, jobs) is their first mandatory argument. No operation at all
defaults to `help`:

    list <namespaces|models|spaces|datasets|jobs>
                          the account's <entity> on every reachable service;
                          each Hub repo is marked loaded/not-loaded (loaded =
                          its submodule is initialized under hf/)
    load <models|spaces|datasets> <namespace> <name>
                          materialize the Hub repo as a git submodule under
                          hf/<namespace>/<entity>/<name>
    unload <models|spaces|datasets> <namespace> <name>
                          deinit the submodule (empties its folder locally;
                          it stays registered and can be loaded again)
    git <models|spaces|datasets> <namespace> <name> <git args...>
                          proxy a git command to that submodule
                          (git -C hf/<namespace>/<entity>/<name> <git args...>)
    execute jobs <namespace>/<model>/<type>/<script_name>
                          submit a runnable to the service that hosts it
    status jobs <run_id> [service]
                          status of one run (service defaults to huggingface)
    create <models|spaces|datasets|jobs> <service> <name>
                          create the entity on the remote service (a service
                          that cannot create it errors out saying so)
    delete <models|spaces|datasets|jobs> <service> <name>
                          delete the entity on the remote service —
                          DESTRUCTIVE and remote-only (local submodules are
                          removed via `unload`, not here)
    help [operation [entity]]
                          usage overview, or one operation's detailed usage
                          (e.g. `help status jobs`) — needs no .env
    (every `list` asks each service for that entity; a service without the
    concept — e.g. Spaces on Kaggle — returns nothing and is skipped; the
    Hub groups repo kinds per account namespace)

Every operation follows one grammar:

    <operation> <entity> [...<mandatory>] [...<optional>] [...<vargs>]

(`help` is the one entity-less operation.) The implementations live in the
lib/ package, one module per concern (see lib/__init__.py); this file only
holds the SPECS table — one declarative entry per operation — and the
generic parser/dispatcher it drives.
Credentials are not checked upfront: every public service method calls its
service's login() first (the BaseService contract), which mandates that
service's credentials as env vars from the workspace .env (template:
.env.example) and exits naming what is missing.

Steps hosted here:
  2. parse           — match argv against the grammar via SPECS (none at all
                       defaults to `help`): mandatory arguments are counted
                       and parsed (e.g. the `jobs execute` target), optional
                       ones get defaults, variadic tails (the git proxies,
                       help topics) are collected verbatim.
  3. dispatch        — hand the parsed arguments to the operation's lib/
                       implementation:
                       jobs.py resolves runnables and submits them to their
                       service in-process (every service is a BaseService
                       subclass in its own service.py — hf/service.py for
                       huggingface, <service>/service.py for
                       externals — imported dynamically); listings.py sweeps
                       the services; repos.py loads/unloads/gits the hf/
                       submodules; helptext.py prints usage.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # lib/ -> workspace root
sys.path.insert(0, str(ROOT))          # so lib/ and services can import baseservice

from dotenv import load_dotenv         # noqa: E402

# The workspace .env (template: .env.example) — credentials the services'
# login flows resolve from the environment. Loaded once here at startup,
# BEFORE the ops imports, so every module sees the same environment.
load_dotenv(ROOT / ".env")

from lib.utilities import show_help, usage_exit             # noqa: E402
from lib.operations import (CreateOperation, DeleteOperation,  # noqa: E402
                            ExecuteOperation, GitOperation, ListOperation,
                            LoadOperation, StatusOperation, UnloadOperation)


# ---------------------------------------------------------------------------
# Steps 2+3 — one grammar for every operation:
#
#   <operation> <entity> [...<mandatory>] [...<optional>] [...<vargs>]
#
# Operations are the verbs (list, load, unload, git, execute, status,
# create, delete); the
# ENTITY they act on (namespaces, models, spaces, datasets, jobs) is their
# FIRST mandatory argument — validated inside each operation's _run like
# every other argument. SPECS holds one BaseOperation instance per verb;
# each instance declares its own argument shape (mandatory_count /
# optional_max) which main uses to slice argv, and BaseOperation.run turns
# any error into error text + that operation's helptext. `help` is not an operation — main handles it via
# lib.utilities.show_help. Adding an operation = one class extending
# BaseOperation (carrying its own _helptext entries) + one SPECS entry here.
# ---------------------------------------------------------------------------
SPECS = {
    "list": ListOperation(),
    "load": LoadOperation(),
    "unload": UnloadOperation(),
    "git": GitOperation(),
    "execute": ExecuteOperation(),
    "status": StatusOperation(),
    "create": CreateOperation(),
    "delete": DeleteOperation(),
}


def main():
    # Drop only LEADING empties (launchers once padded argv) — never filter
    # the middle: git-proxy arguments must reach git verbatim, and something
    # like `commit --allow-empty-message -m ""` legitimately contains an
    # empty arg.
    words = list(sys.argv[1:])
    while words and not words[0]:
        words.pop(0)

    # No operation, or `help [...]` -> the help system (not an operation).
    if not words or words[0] == "help":
        return show_help(words[1:] if words else [], SPECS.values())
    verb, given = words[0], words[1:]
    if verb not in SPECS:
        usage_exit(f"Unknown operation: {verb!r}", SPECS.values())
    op = SPECS[verb]

    # Slice the argument groups by the operation's declared shape
    # (mandatory_count / optional_max — see BaseOperation); each operation
    # VALIDATES them in its _run, and BaseOperation.run turns any error into
    # error text + that operation's helptext.
    mandatory = given[:op.mandatory_count]
    rest = given[op.mandatory_count:]
    if op.optional_max is None:
        optional, vargs = rest, ()         # no vargs: every extra is optional
    else:
        optional = rest[:op.optional_max]
        vargs = tuple(rest[op.optional_max:])
    op.run(mandatory, optional, vargs)


if __name__ == "__main__":
    main()
