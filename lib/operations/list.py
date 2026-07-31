"""`list <entity>` — THE lister, sweeping every reachable service."""

from lib.operations.baseoperation import BaseOperation
from lib.config import HF_DIR, REPO_KINDS, ROOT
from lib.utilities import load_service

# entity word -> the BaseService list(kind) string it maps to
KIND_BY_ENTITY = {"namespaces": "namespaces", "models": "models",
                  "spaces": "spaces", "datasets": "datasets", "jobs": "runs"}


def reachable_services():
    """Yield (name, instance) for every registered service — huggingface
    (hf/service.py) first, then each external that ships a service.py —
    that has usable credentials; the rest are noted and skipped, so one
    missing login never fails a whole listing."""
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


def loaded_repos(kind):
    """"<namespace>/<name>" ids of the <kind> repos loaded locally — i.e.
    whose submodule at hf/<namespace>/<kind>/<name> is initialized.

    A registered-but-uninitialized submodule leaves an EMPTY directory
    behind, so "loaded" means the repo folder has content.
    """
    loaded = set()
    if HF_DIR.is_dir():
        for ns_dir in HF_DIR.iterdir():
            kind_dir = ns_dir / kind
            if not kind_dir.is_dir():
                continue
            for repo in kind_dir.iterdir():
                if repo.is_dir() and any(repo.iterdir()):
                    loaded.add(f"{ns_dir.name}/{repo.name}")
    return loaded


class ListOperation(BaseOperation):
    """`list <entity>` — THE lister (entities: namespaces, models, spaces,
    datasets, jobs): ask each reachable service for the entity's list(kind).
    A service returns None for a kind it has no concept of (e.g. Kaggle has
    no Spaces) and is silently skipped; an empty listing prints as such.

    For the repo entities (models/spaces/datasets), each repo in the Hub's
    listing is marked loaded/not-loaded — loaded meaning its submodule is
    initialized at hf/<namespace>/<entity>/<name>. Only the huggingface
    listing gets markers: repos on other services (e.g. Kaggle datasets)
    are not loadable under hf/ at all."""

    mandatory_count = 1                    # <entity>
    optional_max = None                    # no vargs

    _helptext = [
        ("list namespaces",
         "List the account's namespaces on every service that has them (on\n"
         "the Hub: the user + every org)."),
        ("list models",
         "List the account's models on every service with credentials in .env\n"
         "(the Hub groups them per account namespace, marking each repo loaded\n"
         "or not — loaded = its submodule is initialized under hf/). A service\n"
         "without the concept of models — e.g. Kaggle — is skipped."),
        ("list spaces",
         "List the account's Spaces on every service with credentials in .env\n"
         "(the Hub groups them per account namespace, marking each repo loaded\n"
         "or not — loaded = its submodule is initialized under hf/). A service\n"
         "without the concept of Spaces — e.g. Kaggle — is skipped."),
        ("list datasets",
         "List the account's datasets on every service with credentials in .env\n"
         "(the Hub groups them per account namespace, marking each repo loaded\n"
         "or not — loaded = its submodule is initialized under hf/; Kaggle\n"
         "lists its own). A service without the concept of datasets is skipped."),
        ("list jobs",
         "List the account's runs on every service with credentials in .env\n"
         "(Hugging Face Jobs, plus each external service), newest first, as:\n"
         "id, status, title. Services without credentials are noted and\n"
         "skipped."),
    ]

    def _run(self, mandatory, optional, vargs):
        if len(mandatory) != 1 or optional or vargs:
            raise ValueError("`list` expects exactly one entity")
        entity = mandatory[0]
        if entity not in KIND_BY_ENTITY:
            raise ValueError(f"unknown entity '{entity}' — expected: "
                             f"{' | '.join(KIND_BY_ENTITY)}")
        kind = KIND_BY_ENTITY[entity]
        loaded = loaded_repos(kind) if kind in REPO_KINDS else None
        for name, service in reachable_services():
            listing = service.list(kind)
            if listing is None:
                continue                   # kind unknown to this service
            # Header uses the ENTITY word the user typed (e.g. "jobs"), not
            # the internal list(kind) string it maps to (e.g. "runs").
            print(f"[{name}] {entity}:")
            print_listing(listing, loaded if name == "huggingface" else None)


def print_listing(listing, loaded, indent="  "):
    """Render one service's listing, whatever its shape: grouped dicts (the
    Hub's {namespace: [ids]}) recurse a level; run entries print their
    id/status/title; plain items print as-is — with a loaded/not-loaded
    marker when a loaded-id set is given (repo kinds on the Hub)."""
    if isinstance(listing, dict):
        for group, items in listing.items():
            print(f"{indent}[{group}] ({len(items)}):")
            print_listing(items, loaded, indent + "  ")
        return
    if not listing:
        print(f"{indent}(none)")
    for item in listing:
        if isinstance(item, dict):
            print(f"{indent}{item.get('id')}  {item.get('status')}  {item.get('title')}")
        elif loaded is not None:
            print(f"{indent}{item}  [{'loaded' if item in loaded else 'not loaded'}]")
        else:
            print(f"{indent}{item}")
