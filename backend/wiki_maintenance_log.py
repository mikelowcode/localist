"""
Wiki maintenance audit log.
============================
Append-only, timestamped record of destructive wiki maintenance actions
(currently: orphan removals from document_index, performed by
MemoryManager.reconcile_wiki()).

Distinct in purpose from:
  - application logging (ephemeral, console-only, via the standard logger)
  - sessions-log.md (hand/Claude-Code-narrated dev journal, not a runtime
    artifact)

This is a plain data file — no logging framework, no rotation, no size cap.
If the file ever needs rotation/retention at larger project scale, that is
a forward-looking concern, not handled here.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_PATH = _PROJECT_ROOT / "logs" / "wiki_maintenance.log"


def log_orphan_removed(name: str, path: str) -> None:
    """
    Append one line recording an orphaned document_index row removal.

    Line format (tab-separated):
        <ISO 8601 UTC timestamp>\torphan_removed\tname=<name>\tpath=<path>

    A write failure here must never break the caller's reconciliation flow —
    any exception is caught, logged as a warning, and swallowed.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{timestamp}\torphan_removed\tname={name}\tpath={path}\n"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        f = open(_LOG_PATH, mode="a", encoding="utf-8")
        try:
            f.write(line)
            f.flush()
        finally:
            f.close()
    except Exception as exc:
        logger.warning("wiki_maintenance_log: failed to write orphan-removed entry — %s", exc)
