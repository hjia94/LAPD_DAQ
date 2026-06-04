"""Small logging-teardown helpers shared by the acquire and offload entry scripts.

Both long-running processes write a log file (``motor.log`` for acquire,
``offload.log`` in the spool for offload) and then sit at a pause prompt before
their console closes. A still-open log handle keeps the OS file lock alive, which
on Windows blocks a later *restart* from deleting the HDF5 or renaming the spool
folder. Closing the file handlers on exit (interrupt, error, or normal) releases
those locks, so the cleanup belongs in every process's teardown path.
"""

import logging


def close_log_file_handlers(*loggers):
    """Flush, close, and detach every FileHandler on the given loggers.

    Pass the specific loggers a process configured (and/or the root logger via
    ``logging.getLogger()``). Best-effort: a handler that fails to close is
    skipped so teardown never raises. After this call the underlying log files
    are no longer held open by this process.
    """
    for logger in loggers:
        for handler in list(logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            try:
                handler.flush()
                handler.close()
            except Exception:
                pass
            finally:
                logger.removeHandler(handler)
