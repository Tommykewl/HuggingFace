"""baseoperation.py — the interface every operation implements.

An *operation* is one verb of the launcher grammar
(`<operation> <entity> [...mandatory] [...optional] [...vargs]`) — list,
load, unload, git, execute, status. The ENTITY it acts on (namespaces,
models, spaces, datasets, jobs) is its first mandatory argument, validated
inside _run like every other argument. Each operation is a class extending
BaseOperation — the class name says which verb it is — instantiated once in
lib/main.py's SPECS table and invoked through run().

The contract:

  * mandatory_count (abstract variable, public)
        How many mandatory arguments the operation takes — the entity
        included. main.py slices exactly this many words off argv (after
        the verb) into `mandatory`.
  * optional_max (abstract variable, public)
        The maximum number of optional arguments, or None. Only needed for
        operations that take vargs: with None, EVERYTHING after the
        mandatory arguments is passed as optional arguments (no vargs);
        with a value N, the first N extra arguments are optional and any
        remainder is passed as the vargs tail (e.g. the git proxy uses 0 —
        every trailing argument is a varg).
  * _helptext (abstract variable, protected)
        The operation's help entries: a list of (usage line, detailed help)
        tuples, one per entity the operation acts on. Lives IN the
        operation class (lib/operations/) — lib/utilities.py only aggregates these lists for
        the overview and `help` topics.
  * _run(mandatory, optional, vargs) (abstract, protected)
        The operation itself. Receives the argument groups exactly as the
        grammar defines them: the list of mandatory arguments, the list of
        optional arguments (defaults already filled in), and the tuple of
        trailing varargs. Validates its own arguments — raise (or sys.exit
        with a message) on anything malformed.
  * run(mandatory, optional, vargs) (public)
        The only entry point: a try/except wrapper around _run. ANY error —
        a validation raise, a missing argument surfacing as an unpack
        error, a service failure — is printed together with the operation's
        helptext (separated by a blank line), so a misused operation always
        answers with how to use it. Integer SystemExits pass through
        untouched (the git proxy exits with git's own exit code).
"""

import sys
from abc import ABC, abstractmethod


class BaseOperation(ABC):
    """Interface + error wrapper for launcher operations."""

    # -- contract ------------------------------------------------------------
    @property
    @abstractmethod
    def mandatory_count(self) -> int:
        """Number of mandatory arguments (the entity included).

        Declared abstract so every operation MUST state its count; concrete
        operations override it with a plain class attribute."""

    @property
    @abstractmethod
    def optional_max(self):
        """Maximum number of optional arguments, or None.

        None -> no vargs: everything after the mandatory arguments is
        optional. A value N -> the first N extras are optional, the rest are
        the vargs tail. Overridden with a plain class attribute."""

    @property
    @abstractmethod
    def _helptext(self) -> list:
        """This operation's help entries: [(usage, detail), ...] — one per
        entity. Overridden with a plain class-attribute list."""

    @abstractmethod
    def _run(self, mandatory: list, optional: list, vargs: tuple):
        """The operation body; validates its own arguments."""

    # -- public entry point ---------------------------------------------------
    def help_entries(self) -> list:
        """Public accessor for the (usage, detail) help entries."""
        return list(self._helptext)

    def _helptext_str(self) -> str:
        """The help entries formatted as one printable text."""
        return "\n\n".join(f"Usage: {usage}\n\n{detail}"
                           for usage, detail in self._helptext)

    def run(self, mandatory=(), optional=(), vargs=()):
        """Invoke the operation; on ANY error print it with the helptext."""
        try:
            return self._run(list(mandatory), list(optional), tuple(vargs))
        except SystemExit as exc:
            if isinstance(exc.code, str):  # message exits get the helptext too
                sys.exit(f"{exc.code}\n\n{self._helptext_str()}")
            raise                          # int exit codes (git proxy) untouched
        except Exception as exc:
            sys.exit(f"ERROR: {exc}\n\n{self._helptext_str()}")
