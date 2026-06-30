"""
session_files.py — Ephemeral per-session file cache for LORA chat attachments.

Stores uploaded text file content in a process-lifetime dict. Content is
cleared on backend restart. No persistence, no wiki ingestion, no embedding.

Public API
----------
add_file(filename, content) -> None | str
    Add a file to the cache. Returns None on success, or an error string
    if the per-file ceiling is exceeded or the total slot budget would be
    exceeded by adding this file. Callers should surface the error string
    to the user verbatim — it is user-readable.

remove_file(filename) -> bool
    Remove a named file. Returns True if removed, False if not found.

get_files() -> list[SessionFile]
    Return all cached files in insertion order for prompt assembly.

clear() -> None
    Remove all cached files. Called on explicit clear-chat (future).

Constants
---------
ALLOWED_EXTENSIONS : frozenset[str]
    Client-side allowlist is defence-in-depth only; this is the real gate.
MAX_FILE_TOKENS    : int   4,000 tokens per file (16,000 chars)
MAX_TOTAL_TOKENS   : int   20,000 tokens total across all files
"""

from __future__ import annotations

from collections import OrderedDict

from prompt_builder import SessionFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".txt", ".py", ".ts", ".js", ".svelte", ".json",
    ".yaml", ".yml", ".toml", ".sh", ".env", ".csv", ".xml",
    ".html", ".css", ".rs", ".go", ".rb", ".java", ".c", ".cpp",
    ".h", ".hpp", ".sql",
})

MAX_FILE_TOKENS:  int = 4_000
MAX_TOTAL_TOKENS: int = 20_000

_CHARS_PER_TOKEN: int = 4   # consistent with PromptBuilder._estimate_tokens()

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# OrderedDict preserves insertion order; filename is the key.
_cache: OrderedDict[str, str] = OrderedDict()


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _total_tokens() -> int:
    return sum(_estimate_tokens(content) for content in _cache.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_file(filename: str, content: str) -> str | None:
    """
    Add a file to the cache.

    Returns None on success.
    Returns a user-readable error string on rejection — never raises.

    Rejection conditions (in check order):
      1. File extension not in ALLOWED_EXTENSIONS.
      2. Content exceeds MAX_FILE_TOKENS (per-file ceiling).
      3. Adding this file would exceed MAX_TOTAL_TOKENS (total ceiling).

    If a file with the same filename already exists it is replaced in-place
    (order preserved, total-budget check applied to the new content).
    """
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"File type '{ext}' is not supported. "
            f"Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    file_tokens = _estimate_tokens(content)
    if file_tokens > MAX_FILE_TOKENS:
        return (
            f"'{filename}' is too large ({file_tokens:,} tokens estimated). "
            f"Maximum per file is {MAX_FILE_TOKENS:,} tokens "
            f"({MAX_FILE_TOKENS * _CHARS_PER_TOKEN:,} characters)."
        )

    # Calculate budget excluding any existing entry for this filename
    existing_tokens = _estimate_tokens(_cache[filename]) if filename in _cache else 0
    projected_total = _total_tokens() - existing_tokens + file_tokens
    if projected_total > MAX_TOTAL_TOKENS:
        return (
            f"Adding '{filename}' would exceed the session file budget "
            f"({projected_total:,} tokens projected, limit {MAX_TOTAL_TOKENS:,}). "
            f"Remove a file first."
        )

    _cache[filename] = content
    return None


def remove_file(filename: str) -> bool:
    """Remove a named file. Returns True if removed, False if not found."""
    if filename in _cache:
        del _cache[filename]
        return True
    return False


def get_files() -> list[SessionFile]:
    """Return all cached files in insertion order."""
    return [SessionFile(filename=k, content=v) for k, v in _cache.items()]


def clear() -> None:
    """Remove all cached files."""
    _cache.clear()
