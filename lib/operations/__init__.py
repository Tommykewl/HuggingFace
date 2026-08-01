"""lib.operations — the BaseOperation interface plus one module per
operation class (the verbs of the launcher grammar). Each class declares
its argument shape (mandatory_count / optional_max) and carries its own
helptext: the _helptext list of (usage, detail) entries, one per entity it
acts on.

  baseoperation.py BaseOperation — the interface + error wrapper (run()
              prints any error together with the operation's helptext)
  list.py     ListOperation    — list <namespaces|models|spaces|datasets|jobs>
  repo.py     RepoOperation    — shared base for the repo operations below
  load.py     LoadOperation    — load <models|spaces|datasets> <ns> <name>
  unload.py   UnloadOperation  — unload <models|spaces|datasets> <ns> <name>
  git.py      GitOperation     — git <models|spaces|datasets> <ns> <name> <args...>
  execute.py  ExecuteOperation — execute jobs <ns>/<model>/<type>/<script>
  status.py   StatusOperation  — status jobs <run_id> [service]
  remote.py   RemoteOperation  — shared base for the remote-entity operations below
  create.py   CreateOperation  — create <models|spaces|datasets|jobs> <service> <name>
  delete.py   DeleteOperation  — delete <models|spaces|datasets|jobs> <service> <name>
"""

from lib.operations.create import CreateOperation
from lib.operations.delete import DeleteOperation
from lib.operations.execute import ExecuteOperation
from lib.operations.git import GitOperation
from lib.operations.list import ListOperation
from lib.operations.load import LoadOperation
from lib.operations.status import StatusOperation
from lib.operations.unload import UnloadOperation
