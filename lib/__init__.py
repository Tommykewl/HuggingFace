"""lib — the workspace operations, one module per concern (the mlops.sh /
mlops.ps1 launchers delegate to main.py):

  main.py         the entrypoint: the SPECS table of operation instances +
                  the generic parser/dispatcher (every operation is
                  <operation> <entity> [...mandatory] [...optional] [...vargs])
  baseservice.py  the BaseService interface every service runner implements
                  (hf/service.py, kaggle/service.py, ...)
  config.py       workspace-wide constants (paths, types, repo kinds)
  utilities.py    shared utilities in three sections: the help aggregator
                  (helptexts live IN the operation classes), service loading
                  (BaseService runners, in-process), and git plumbing (the
                  single git executor + registered-submodule query)
  operations/     the BaseOperation interface + one module per operation
                  class (list, load, unload, git, execute, status)
"""
