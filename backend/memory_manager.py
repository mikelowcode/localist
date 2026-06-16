"""
LORA — SQLite MemoryManager
============================
Persistent, local-first replacement for the in-process MemoryManager
currently defined in controller_agent.py.

Layer placement
---------------
  ControllerAgent / ResearchAgent  →  MemoryManager  →  SQLite (local file)
                                                       →  EmbeddingGemma (optional)

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no agent logic.
- Drop-in replacement for the in-process MemoryManager in controller_agent.py.
  The ControllerAgent API (add / add_agent_result / format_for_prompt /
  get_context_window / clear) is fully preserved so no call sites change.
- ResearchAgent gains a new retrieval path: query_corpus() replaces the
  full filesystem walk in _load_corpus() when a MemoryManager is available.
- All data lives in a single SQLite file.  No external services.  The file
  persists across server restarts, page refreshes, and process kills.
- Embeddings are optional.  When an embed callable is not supplied, keyword
  overlap scoring is used for retrieval (same strategy as the current
  ResearchAgent).  Embeddings can be enabled at any time without a schema
  migration — the column is always present, just NULL until populated.
- Thread-safe: every public method acquires a threading.Lock before any
  DB write.  Reads use a separate connection per call (WAL mode allows
  concurrent readers alongside one writer).

Database file location
----------------------
Defaults to  <project_root>/lora_memory.db
Override via the ``db_path`` constructor argument or the LOCALIST_MEMORY_DB
environment variable (the latter is read by main.py's Settings class —
add  memory_db: str | None = None  to Settings and pass it through).

Schema — three tables
----------------------

  conversation_log
  ----------------
  Append-only log of every agent turn within a task session.
  Mirrors the existing in-process MemoryManager._entries list but survives
  restarts.  Used by Synthesizer.format_for_prompt().

    id          INTEGER PRIMARY KEY AUTOINCREMENT
    task_id     TEXT    NOT NULL          — groups entries by task
    role        TEXT    NOT NULL          — "user" | "agent" | "system"
    content     TEXT    NOT NULL
    meta_json   TEXT    DEFAULT '{}'      — JSON-encoded metadata dict
    created_at  REAL    NOT NULL          — unix timestamp (time.time())

  document_index
  --------------
  One row per unique document (wiki page or raw file).  Updated whenever
  WikiAgent writes a new page.  ResearchAgent queries this table instead
  of walking the filesystem on every call.

    id           INTEGER PRIMARY KEY AUTOINCREMENT
    name         TEXT    NOT NULL          — page stem (kebab-case)
    path         TEXT    NOT NULL UNIQUE   — absolute path on disk
    doc_type     TEXT    NOT NULL          — "wiki" | "raw"
    content      TEXT    NOT NULL          — full UTF-8 text
    token_set    TEXT    NOT NULL DEFAULT '' — space-separated lowercase tokens
                                             (pre-computed for fast keyword scoring)
    embedding    BLOB    DEFAULT NULL      — packed float32 array (struct.pack)
                                             NULL until embed() is called
    content_hash TEXT    NOT NULL DEFAULT '' — sha256[:16] for change detection
    indexed_at   REAL    NOT NULL          — unix timestamp of last index/update

  retrieval_cache
  ---------------
  Optional query-level cache.  Maps a query string + top-N to a JSON array
  of (name, score) pairs.  Invalidated whenever document_index changes.
  Cheap hit rate gain for repeated identical sub-queries within ResearchAgent.

    id          INTEGER PRIMARY KEY AUTOINCREMENT
    query_hash  TEXT    NOT NULL          — sha256 of (query + str(top_n))
    top_n       INTEGER NOT NULL
    result_json TEXT    NOT NULL          — JSON: [{name, path, doc_type, score}]
    created_at  REAL    NOT NULL
    valid       INTEGER NOT NULL DEFAULT 1  — 0 = invalidated on index mutation

Embedding storage
-----------------
Embeddings are stored as raw bytes using struct.pack(f">{n}f", *vector).
Big-endian float32, one float per dimension (768 floats × 4 bytes = 3072 bytes).
This is faster to serialise/deserialise than JSON and avoids numpy/sqlite3
adapter complexity.  Cosine similarity is computed in pure Python on retrieval
(same as the existing _cosine_similarity helper) — fine for corpora up to
a few thousand documents.  At much larger scale, swap in a sqlite-vec extension
or a dedicated vector store without changing the public API.

Integration points
------------------
1. main.py lifespan — construct MemoryManager once, store on _state, pass to
   agents that need it:

     from memory_manager import MemoryManager
     memory_manager = MemoryManager(db_path=..., embed_fn=runtime.embed)
     _state.memory_manager = memory_manager

2. ControllerAgent — replace the per-request in-process MemoryManager with the
   persistent one.  Pass it into the controller constructor:

     controller = ControllerAgent(
         runtime        = runtime,
         agents         = [wiki_agent, research_agent],
         memory_manager = memory_manager,   # new optional param
     )

   Inside ControllerAgent._execute(), replace:
     memory = MemoryManager()
   with:
     memory = self._memory_manager or MemoryManager()

3. WikiAgent — after writing a page, call:
     memory_manager.index_document(path, doc_type="wiki", embed=True)
   This keeps the index current without a full corpus reload.

4. ResearchAgent — replace _load_corpus() with:
     docs = memory_manager.query_corpus(query, max_results=max_src)
   Pass use_embeddings through context as before; the manager handles both
   keyword and embedding retrieval transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION   = 2          # increment when schema changes require migration
_EMBEDDING_DIM    = 768        # EmbeddingGemma-300M-4bit output dimension
_EMBEDDING_FORMAT = ">768f"    # big-endian float32 × 768

# Soft cap: keep at most this many conversation_log rows per task_id.
# Older rows are deleted (FIFO) when the cap is exceeded.
_CONV_LOG_CAP_PER_TASK = 200


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _pack_embedding(vector: list[float]) -> bytes:
    """Pack a float list into big-endian float32 bytes."""
    return struct.pack(f">{len(vector)}f", *vector)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack big-endian float32 bytes into a float list."""
    n = len(blob) // 4
    return list(struct.unpack(f">{n}f", blob))


def _tokenize(text: str) -> set[str]:
    """Word-level token set for keyword overlap scoring."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_score(query_tokens: set[str], token_set_str: str) -> float:
    """Jaccard-like overlap between query tokens and a pre-computed token set."""
    doc_tokens = set(token_set_str.split()) if token_set_str else set()
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _content_hash(text: str) -> str:
    """First 16 hex chars of the SHA-256 digest — used for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _query_hash(query: str, top_n: int) -> str:
    """Cache key for the retrieval_cache table."""
    raw = f"{query}||{top_n}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Document result type (mirrors _Document in research_agent.py)
# ---------------------------------------------------------------------------

class DocumentResult:
    """
    A document returned by query_corpus().

    Mirrors the public interface of research_agent._Document so that
    ResearchAgent can use either path without changing its internal logic.
    The to_source_dict() method matches the existing shape consumed by the
    Synthesizer's _collect_sources() helper.
    """

    __slots__ = ("name", "path", "doc_type", "content", "relevance_score")

    def __init__(
        self,
        name:            str,
        path:            Path,
        doc_type:        str,
        content:         str,
        relevance_score: float = 0.0,
    ) -> None:
        self.name            = name
        self.path            = path
        self.doc_type        = doc_type
        self.content         = content
        self.relevance_score = relevance_score

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "name":            self.name,
            "path":            str(self.path),
            "type":            self.doc_type,
            "relevance_score": round(self.relevance_score, 4),
        }


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    SQLite-backed persistent memory for LORA.

    Responsibilities
    ----------------
    1. Conversation log  — persist and retrieve per-task agent turns.
    2. Document index    — index wiki pages and raw files for fast retrieval.
    3. Corpus retrieval  — return ranked DocumentResults for a query without
                          walking the filesystem.
    4. Embedding cache   — store and reuse 768-dim EmbeddingGemma vectors.
    5. Retrieval cache   — cache query→results mappings, invalidated on writes.

    Parameters
    ----------
    db_path :
        Path to the SQLite file.  Created (with parent dirs) if absent.
        Defaults to  <this file's parent>/lora_memory.db.
    embed_fn :
        Optional callable that accepts a text string and returns a list[float]
        of length 768.  When provided, embeddings are computed and stored on
        index_document() and used for cosine re-ranking in query_corpus().
        When absent, keyword overlap scoring is used exclusively.
    """

    def __init__(
        self,
        db_path:  Path | str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "lora_memory.db"
        self._db_path  = Path(db_path)
        self._embed_fn = embed_fn
        self._lock     = threading.Lock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        logger.info(
            "MemoryManager initialised — db=%s  embeddings=%s",
            self._db_path,
            "enabled" if embed_fn else "disabled (keyword-only)",
        )

    # -----------------------------------------------------------------------
    # Database initialisation
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """
        Open a new connection to the database.

        WAL mode allows concurrent reads while a write is in progress.
        foreign_keys and journal_mode are set on every connection.
        """
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")   # safe with WAL; faster than FULL
        return conn

    def _init_db(self) -> None:
        """
        Create tables and indexes if they do not already exist.

        The schema_version table holds a single row.  If the stored version
        is less than _SCHEMA_VERSION, _migrate() is called before returning.
        This is the extension point for future schema changes.
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version  INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS conversation_log (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id     TEXT    NOT NULL,
                        role        TEXT    NOT NULL,
                        content     TEXT    NOT NULL,
                        meta_json   TEXT    NOT NULL DEFAULT '{}',
                        created_at  REAL    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_conv_task
                        ON conversation_log(task_id, created_at);

                    CREATE TABLE IF NOT EXISTS document_index (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        name         TEXT    NOT NULL,
                        path         TEXT    NOT NULL UNIQUE,
                        doc_type     TEXT    NOT NULL,
                        content      TEXT    NOT NULL,
                        token_set    TEXT    NOT NULL DEFAULT '',
                        embedding    BLOB    DEFAULT NULL,
                        content_hash TEXT    NOT NULL DEFAULT '',
                        indexed_at   REAL    NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_doc_name
                        ON document_index(name);
                    CREATE INDEX IF NOT EXISTS idx_doc_type
                        ON document_index(doc_type);

                    CREATE TABLE IF NOT EXISTS retrieval_cache (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_hash  TEXT    NOT NULL UNIQUE,
                        top_n       INTEGER NOT NULL,
                        result_json TEXT    NOT NULL,
                        created_at  REAL    NOT NULL,
                        valid       INTEGER NOT NULL DEFAULT 1
                    );
                    CREATE INDEX IF NOT EXISTS idx_cache_hash
                        ON retrieval_cache(query_hash, valid);

                    CREATE TABLE IF NOT EXISTS episodes (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_type    TEXT    NOT NULL,
                        subject         TEXT    NOT NULL,
                        content         TEXT    NOT NULL,
                        confidence      REAL    NOT NULL DEFAULT 1.0,
                        source          TEXT    NOT NULL,
                        task_id         TEXT,
                        conversation_id TEXT,
                        project_context TEXT,
                        status          TEXT    NOT NULL DEFAULT 'active',
                        created_at      REAL    NOT NULL,
                        last_accessed   REAL,
                        embedding       BLOB
                    );
                    CREATE INDEX IF NOT EXISTS idx_episodes_type_status
                        ON episodes (episode_type, status);
                    CREATE INDEX IF NOT EXISTS idx_episodes_subject
                        ON episodes (subject, status);
                    CREATE INDEX IF NOT EXISTS idx_episodes_project
                        ON episodes (project_context, status);
                """)

                # Ensure schema_version row exists.
                row = conn.execute("SELECT version FROM schema_version").fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (_SCHEMA_VERSION,),
                    )
                    conn.commit()
                    logger.debug("schema_version initialised to %d.", _SCHEMA_VERSION)
                elif row["version"] < _SCHEMA_VERSION:
                    self._migrate(conn, from_version=row["version"])

            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """
        Apply incremental schema migrations.

        Called only when the stored schema_version < _SCHEMA_VERSION.
        Add elif blocks here for each new version.
        """
        logger.info(
            "Migrating MemoryManager schema from v%d to v%d.",
            from_version,
            _SCHEMA_VERSION,
        )

        if from_version < 2:
            logger.info("Applying migration v1→v2: creating episodes table.")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_type    TEXT    NOT NULL,
                    subject         TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    confidence      REAL    NOT NULL DEFAULT 1.0,
                    source          TEXT    NOT NULL,
                    task_id         TEXT,
                    conversation_id TEXT,
                    project_context TEXT,
                    status          TEXT    NOT NULL DEFAULT 'active',
                    created_at      REAL    NOT NULL,
                    last_accessed   REAL,
                    embedding       BLOB
                );
                CREATE INDEX IF NOT EXISTS idx_episodes_type_status
                    ON episodes (episode_type, status);
                CREATE INDEX IF NOT EXISTS idx_episodes_subject
                    ON episodes (subject, status);
                CREATE INDEX IF NOT EXISTS idx_episodes_project
                    ON episodes (project_context, status);
            """)

        conn.execute(
            "UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,)
        )
        conn.commit()
        logger.info("Migration complete. Schema is now v%d.", _SCHEMA_VERSION)

    # -----------------------------------------------------------------------
    # Conversation log  (ControllerAgent / Synthesizer API)
    # -----------------------------------------------------------------------

    def add(
        self,
        role:     str,
        content:  str,
        metadata: dict[str, Any] | None = None,
        task_id:  str = "global",
    ) -> None:
        """
        Append one entry to the conversation log.

        Parameters
        ----------
        role :
            "user" | "agent" | "system"
        content :
            The text content of the entry.
        metadata :
            Optional dict stored as JSON.  Useful for agent name, subtask_id, etc.
        task_id :
            Groups entries by task.  Use the task's UUID.  Defaults to "global"
            for entries that are not task-scoped (e.g. system messages).
        """
        meta_json = json.dumps(metadata or {})
        now       = time.time()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO conversation_log
                        (task_id, role, content, meta_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (task_id, role, content, meta_json, now),
                )
                conn.commit()
                self._evict_conversation_log(conn, task_id)
            finally:
                conn.close()

    def add_agent_result(
        self,
        result:  Any,           # AgentResult — typed Any to avoid circular import
        task_id: str = "global",
    ) -> None:
        """
        Convenience wrapper matching the existing MemoryManager API.

        Serialises result.output as the content string, preserving the same
        format the Synthesizer's format_for_prompt() already expects.
        """
        self.add(
            role     = "agent",
            content  = str(result.output),
            metadata = {
                "agent":      result.agent_name,
                "subtask_id": result.subtask_id,
            },
            task_id  = task_id,
        )

    def get_context_window(
        self,
        task_id:    str        = "global",
        limit:      int        = 50,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent ``limit`` entries for a task, optionally
        capped by a token budget.

        Parameters
        ----------
        task_id :
            Groups entries by task. Defaults to "global".
        limit :
            Maximum number of rows to fetch from the DB before token
            trimming is applied. Defaults to 50.
        max_tokens :
            When provided, entries are trimmed (oldest first) until the
            total estimated token count of all remaining entries is at or
            below this value. Token count is estimated as
            ``len(content) // 4`` per entry (1 token ≈ 4 characters).
            Truncation never cuts mid-entry.
            When None (default), no token trimming is applied and all
            ``limit`` entries are returned as before.

        Returns
        -------
        list[dict[str, Any]]
            Dicts with keys: role, content, metadata.
            Ordered chronologically (oldest first).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT role, content, meta_json
                FROM   conversation_log
                WHERE  task_id = ?
                ORDER  BY created_at DESC
                LIMIT  ?
                """,
                (task_id, limit),
            ).fetchall()
        finally:
            conn.close()

        # Reverse to chronological order (oldest first)
        entries = [
            {
                "role":     row["role"],
                "content":  row["content"],
                "metadata": json.loads(row["meta_json"]),
            }
            for row in reversed(rows)
        ]

        # Apply token ceiling: drop oldest entries until budget is met
        if max_tokens is not None:
            while entries:
                total_chars = sum(len(e["content"]) for e in entries)
                estimated_tokens = total_chars // 4
                if estimated_tokens <= max_tokens:
                    break
                entries.pop(0)   # drop the oldest entry

        return entries

    def format_for_prompt(
        self,
        task_id: str = "global",
        limit:   int = 50,
    ) -> str:
        """
        Flatten the conversation log into a single prompt-ready string.

        Matches the existing MemoryManager.format_for_prompt() output exactly
        so the Synthesizer needs no changes.
        """
        entries = self.get_context_window(task_id=task_id, limit=limit)
        return "\n".join(
            f"[{e['role'].upper()}] {e['content']}" for e in entries
        )

    def clear(self, task_id: str | None = None) -> None:
        """
        Delete conversation log entries.

        If task_id is given, only entries for that task are deleted.
        If task_id is None, ALL conversation log entries are deleted.
        The document index is NOT affected by clear().
        """
        with self._lock:
            conn = self._connect()
            try:
                if task_id is not None:
                    conn.execute(
                        "DELETE FROM conversation_log WHERE task_id = ?",
                        (task_id,),
                    )
                    logger.debug("Cleared conversation log for task_id=%s.", task_id)
                else:
                    conn.execute("DELETE FROM conversation_log")
                    logger.debug("Cleared all conversation log entries.")
                conn.commit()
            finally:
                conn.close()

    def _evict_conversation_log(
        self,
        conn:    sqlite3.Connection,
        task_id: str,
    ) -> None:
        """
        Delete the oldest rows for task_id when the per-task cap is exceeded.

        Called inside an existing write transaction — no lock re-entry needed.
        Silently skips if the count is within the cap.
        """
        count = conn.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0]

        if count > _CONV_LOG_CAP_PER_TASK:
            excess = count - _CONV_LOG_CAP_PER_TASK
            conn.execute(
                """
                DELETE FROM conversation_log
                WHERE  id IN (
                    SELECT id FROM conversation_log
                    WHERE  task_id = ?
                    ORDER  BY created_at ASC
                    LIMIT  ?
                )
                """,
                (task_id, excess),
            )
            logger.debug(
                "Evicted %d old conversation_log rows for task_id=%s.",
                excess, task_id,
            )

    # -----------------------------------------------------------------------
    # Document index  (WikiAgent → index; ResearchAgent → query)
    # -----------------------------------------------------------------------

    def index_document(
        self,
        path:        Path | str,
        doc_type:    str,            # "wiki" | "raw"
        content:     str | None = None,
        embed:       bool       = True,
    ) -> None:
        """
        Add or update a document in the index.

        Parameters
        ----------
        path :
            Absolute path to the file on disk.
        doc_type :
            "wiki" for pages in wiki/, "raw" for files in raw/.
        content :
            File text.  If None, the file is read from disk.  Supply it
            directly if you already have the text in memory (e.g. from
            WikiAgent after writing to disk) to avoid a redundant read.
        embed :
            Whether to compute and store an embedding for this document.
            Requires embed_fn to have been supplied at construction time.
            Set False when bulk-indexing many documents and you want to
            embed them in a separate batch pass.

        Behaviour on collision (same path):
            If the content hash is unchanged, the row is left as-is.
            If the content changed, the row is updated and the retrieval
            cache is invalidated.
        """
        path = Path(path).resolve()
        if content is None:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("index_document: cannot read %s — %s", path, exc)
                return

        name         = path.stem
        token_set    = " ".join(sorted(_tokenize(content)))
        c_hash       = _content_hash(content)
        now          = time.time()

        # Compute embedding before acquiring the lock — it can be slow.
        embedding_blob: bytes | None = None
        if embed and self._embed_fn is not None:
            try:
                # Embed the first ~500 chars — consistent with the existing
                # ResearchAgent strategy that keeps embedding calls cheap.
                vec = self._embed_fn(content[:500])
                if len(vec) == _EMBEDDING_DIM:
                    embedding_blob = _pack_embedding(vec)
                else:
                    logger.warning(
                        "index_document: embed returned dim=%d, expected %d — skipping.",
                        len(vec), _EMBEDDING_DIM,
                    )
            except Exception as exc:
                logger.warning("index_document: embed failed for %s — %s", path.name, exc)

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT id, content_hash FROM document_index WHERE path = ?",
                    (str(path),),
                ).fetchone()

                if existing is not None:
                    if existing["content_hash"] == c_hash:
                        logger.debug(
                            "index_document: %s unchanged (hash match), skipping.",
                            path.name,
                        )
                        return
                    # Content changed — update and invalidate cache.
                    conn.execute(
                        """
                        UPDATE document_index
                        SET    name=?, doc_type=?, content=?, token_set=?,
                               embedding=?, content_hash=?, indexed_at=?
                        WHERE  path=?
                        """,
                        (name, doc_type, content, token_set,
                         embedding_blob, c_hash, now, str(path)),
                    )
                    logger.info("index_document: updated %s (%s).", path.name, doc_type)
                else:
                    conn.execute(
                        """
                        INSERT INTO document_index
                            (name, path, doc_type, content, token_set,
                             embedding, content_hash, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (name, str(path), doc_type, content, token_set,
                         embedding_blob, c_hash, now),
                    )
                    logger.info("index_document: indexed %s (%s).", path.name, doc_type)

                conn.commit()
                # Invalidate retrieval cache on any index mutation.
                self._invalidate_cache(conn)
            finally:
                conn.close()

    def index_directory(
        self,
        directory: Path | str,
        doc_type:  str,
        embed:     bool = True,
        extensions: set[str] = frozenset({".md", ".txt"}),
    ) -> int:
        """
        Bulk-index all matching files in a directory.

        Skips files whose content hash matches the stored value (idempotent).
        Returns the number of files newly indexed or updated.

        Parameters
        ----------
        directory :
            Path to walk (non-recursive — top-level files only, matching
            the existing _load_corpus() behaviour).
        doc_type :
            "wiki" or "raw".
        embed :
            Whether to compute embeddings.  Pass False for the first bulk
            import of a large directory and run a separate embed pass later.
        extensions :
            File extensions to include.
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            logger.warning("index_directory: %s does not exist.", directory)
            return 0

        count = 0
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning("index_directory: cannot read %s — %s", p, exc)
                    continue
                self.index_document(p, doc_type=doc_type, content=content, embed=embed)
                count += 1

        logger.info(
            "index_directory: indexed %d files from %s (%s).",
            count, directory, doc_type,
        )
        return count

    def remove_document(self, path: Path | str) -> None:
        """
        Remove a document from the index by path.

        Also invalidates the retrieval cache.
        """
        path = Path(path).resolve()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "DELETE FROM document_index WHERE path = ?", (str(path),)
                )
                conn.commit()
                self._invalidate_cache(conn)
                logger.info("remove_document: removed %s.", path.name)
            finally:
                conn.close()

    # -----------------------------------------------------------------------
    # Corpus retrieval  (ResearchAgent API)
    # -----------------------------------------------------------------------

    def query_corpus(
        self,
        query:          str,
        max_results:    int  = 5,
        doc_type:       str | None = None,   # None = all; "wiki" | "raw" to filter
        use_embeddings: bool = True,
    ) -> list[DocumentResult]:
        """
        Return the top-N most relevant documents for a query.

        Strategy
        --------
        1. Score every indexed document by keyword overlap (Jaccard).
        2. If use_embeddings=True and embed_fn is available, re-rank the
           top 2*max_results keyword candidates by embedding cosine similarity
           and return the top max_results of those.
        3. Fall back to keyword ranking if embedding fails or is unavailable.

        The retrieval_cache is checked first.  On a cache hit (same query +
        max_results and cache is still valid), documents are fetched by name
        from the index rather than re-scoring the whole table.  This makes
        repeated identical sub-queries within a single ResearchAgent run free.

        Parameters
        ----------
        query :
            The sub-query string from ResearchAgent.
        max_results :
            Maximum number of DocumentResults to return.
        doc_type :
            Optional filter.  "wiki" returns only wiki pages; "raw" only
            raw files; None returns both.
        use_embeddings :
            Whether to attempt embedding-based re-ranking.

        Returns
        -------
        list[DocumentResult]
            Sorted by relevance_score descending.
        """
        q_hash = _query_hash(query, max_results)

        # -- Cache check --
        cached = self._check_cache(q_hash)
        if cached is not None:
            logger.debug("query_corpus: cache hit for query '%s...'.", query[:40])
            return self._hydrate_cache_result(cached, doc_type)

        # -- Load all documents from the index (token_set + embedding) --
        conn = self._connect()
        try:
            where_clause = "WHERE doc_type = ?" if doc_type else ""
            params       = (doc_type,) if doc_type else ()
            rows = conn.execute(
                f"""
                SELECT name, path, doc_type, content, token_set, embedding
                FROM   document_index
                {where_clause}
                """,
                params,
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            logger.debug("query_corpus: index is empty — returning [].")
            return []

        query_tokens = _tokenize(query)

        # Score by keyword overlap
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = _keyword_score(query_tokens, row["token_set"])
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Embedding re-rank on top-2N candidates
        if use_embeddings and self._embed_fn is not None:
            pool = scored[: max_results * 2]
            try:
                query_vec = self._embed_fn(query)
                re_scored: list[tuple[float, sqlite3.Row]] = []
                for _, row in pool:
                    if row["embedding"] is not None:
                        doc_vec = _unpack_embedding(row["embedding"])
                        score   = _cosine_similarity(query_vec, doc_vec)
                    else:
                        # Fall back to keyword score for un-embedded docs.
                        score = _keyword_score(query_tokens, row["token_set"])
                    re_scored.append((score, row))
                re_scored.sort(key=lambda x: x[0], reverse=True)
                scored = re_scored
            except Exception as exc:
                logger.warning(
                    "query_corpus: embedding re-rank failed (%s); using keyword scores.", exc
                )

        top = scored[:max_results]

        # Build DocumentResult list
        results = [
            DocumentResult(
                name            = row["name"],
                path            = Path(row["path"]),
                doc_type        = row["doc_type"],
                content         = row["content"],
                relevance_score = score,
            )
            for score, row in top
        ]

        # Write to cache (store name+score pairs — content is re-fetched from index)
        cache_payload = [
            {
                "name":     r.name,
                "path":     str(r.path),
                "doc_type": r.doc_type,
                "score":    r.relevance_score,
            }
            for r in results
        ]
        self._write_cache(q_hash, max_results, cache_payload)

        logger.debug(
            "query_corpus: returning %d results for query '%s...'.",
            len(results), query[:40],
        )
        return results

    def get_all_documents(
        self,
        doc_type: str | None = None,
    ) -> list[DocumentResult]:
        """
        Return every document in the index, unsorted.

        Used by WikiAgent's build_wiki_context() to get the full wiki page
        list without scoring.  Replaces the _load_wiki_pages() filesystem
        walk when a MemoryManager is available.

        Parameters
        ----------
        doc_type :
            Optional filter: "wiki", "raw", or None for all.
        """
        conn = self._connect()
        try:
            if doc_type:
                rows = conn.execute(
                    "SELECT name, path, doc_type, content FROM document_index WHERE doc_type = ?",
                    (doc_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, path, doc_type, content FROM document_index",
                ).fetchall()
        finally:
            conn.close()

        return [
            DocumentResult(
                name     = row["name"],
                path     = Path(row["path"]),
                doc_type = row["doc_type"],
                content  = row["content"],
            )
            for row in rows
        ]

    def document_count(self, doc_type: str | None = None) -> int:
        """Return the number of indexed documents, optionally filtered by type."""
        conn = self._connect()
        try:
            if doc_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM document_index WHERE doc_type = ?",
                    (doc_type,),
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM document_index"
            ).fetchone()[0]
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Retrieval cache helpers
    # -----------------------------------------------------------------------

    def _check_cache(self, q_hash: str) -> list[dict] | None:
        """Return the cached result list if a valid entry exists, else None."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT result_json FROM retrieval_cache
                WHERE  query_hash = ? AND valid = 1
                """,
                (q_hash,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            return None

    def _write_cache(
        self,
        q_hash:  str,
        top_n:   int,
        payload: list[dict],
    ) -> None:
        """Upsert a cache entry (non-blocking — cache misses are harmless)."""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    conn.execute(
                        """
                        INSERT INTO retrieval_cache
                            (query_hash, top_n, result_json, created_at, valid)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(query_hash) DO UPDATE SET
                            result_json = excluded.result_json,
                            created_at  = excluded.created_at,
                            valid       = 1
                        """,
                        (q_hash, top_n, json.dumps(payload), time.time()),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as exc:
            logger.debug("_write_cache: failed (non-fatal) — %s", exc)

    def _invalidate_cache(self, conn: sqlite3.Connection) -> None:
        """
        Mark all retrieval cache entries invalid.

        Called inside an existing write transaction whenever the document
        index changes so stale results are never returned.
        """
        conn.execute("UPDATE retrieval_cache SET valid = 0")

    def _hydrate_cache_result(
        self,
        cached:   list[dict],
        doc_type: str | None,
    ) -> list[DocumentResult]:
        """
        Reconstruct DocumentResults from a cache hit.

        Fetches full content from the index by path (content may have changed
        if the doc was updated after the cache entry was written, but since
        _invalidate_cache() is called on every mutation, a valid cache entry
        always corresponds to current content — the path lookup is safe).
        """
        if not cached:
            return []

        paths   = [r["path"] for r in cached]
        placeholders = ",".join("?" * len(paths))
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT name, path, doc_type, content
                FROM   document_index
                WHERE  path IN ({placeholders})
                """,
                paths,
            ).fetchall()
        finally:
            conn.close()

        # Build a map for O(1) lookup
        row_map = {row["path"]: row for row in rows}

        results: list[DocumentResult] = []
        for entry in cached:
            row = row_map.get(entry["path"])
            if row is None:
                continue   # document was deleted since cache was written
            if doc_type and row["doc_type"] != doc_type:
                continue
            results.append(DocumentResult(
                name            = row["name"],
                path            = Path(row["path"]),
                doc_type        = row["doc_type"],
                content         = row["content"],
                relevance_score = entry["score"],
            ))
        return results

    # -----------------------------------------------------------------------
    # Housekeeping / diagnostics
    # -----------------------------------------------------------------------

    def purge_cache(self) -> None:
        """Delete all retrieval cache entries (valid and invalid)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM retrieval_cache")
                conn.commit()
                logger.info("purge_cache: retrieval cache cleared.")
            finally:
                conn.close()

    def stats(self) -> dict[str, Any]:
        """
        Return a summary dict suitable for the GET /health endpoint.

        Keys
        ----
        db_path         — absolute path to the SQLite file
        db_size_kb      — file size on disk
        wiki_docs       — count of indexed wiki pages
        raw_docs        — count of indexed raw files
        conv_log_rows   — total conversation log entries
        cache_valid     — number of valid retrieval cache entries
        cache_invalid   — number of invalidated cache entries
        embeddings_pct  — percentage of documents with an embedding (0–100)
        """
        conn = self._connect()
        try:
            wiki_count  = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE doc_type='wiki'"
            ).fetchone()[0]
            raw_count   = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE doc_type='raw'"
            ).fetchone()[0]
            conv_count  = conn.execute(
                "SELECT COUNT(*) FROM conversation_log"
            ).fetchone()[0]
            cache_valid = conn.execute(
                "SELECT COUNT(*) FROM retrieval_cache WHERE valid=1"
            ).fetchone()[0]
            cache_inv   = conn.execute(
                "SELECT COUNT(*) FROM retrieval_cache WHERE valid=0"
            ).fetchone()[0]
            total_docs  = wiki_count + raw_count
            embedded    = conn.execute(
                "SELECT COUNT(*) FROM document_index WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        finally:
            conn.close()

        emb_pct = round(100 * embedded / total_docs, 1) if total_docs else 0.0
        db_kb   = round(self._db_path.stat().st_size / 1024, 1) if self._db_path.exists() else 0.0

        return {
            "db_path":       str(self._db_path),
            "db_size_kb":    db_kb,
            "wiki_docs":     wiki_count,
            "raw_docs":      raw_count,
            "conv_log_rows": conv_count,
            "cache_valid":   cache_valid,
            "cache_invalid": cache_inv,
            "embeddings_pct": emb_pct,
        }

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"db={self._db_path.name!r}, "
            f"embeddings={'on' if self._embed_fn else 'off'})"
        )


# ---------------------------------------------------------------------------
# EpisodicMemoryWriter
# ---------------------------------------------------------------------------

# Closed set of valid episode types. Adding a type is an architectural
# decision — do not expand this set without updating LOCALIST-Architecture.md.
VALID_EPISODE_TYPES: frozenset[str] = frozenset({
    "preference",
    "correction",
    "decision",
    "workflow",
    "project_fact",
    "task_completion",
    "naming_convention",
})

VALID_SOURCES: frozenset[str] = frozenset({
    "explicit",
    "model_extracted",
})

VALID_STATUSES: frozenset[str] = frozenset({
    "active",
    "superseded",
    "retracted",
})


class EpisodicMemoryWriter:
    """
    Writes episodes to the `episodes` table in `lora_memory.db`.

    Responsibilities
    ----------------
    - insert()  : Write a new active episode, superseding any existing
                  active record with the same (subject, episode_type).
    - retract() : Mark an active episode as retracted by (subject, episode_type).
    - _supersede_existing() : Internal helper — marks conflicting active
                              records as superseded before a new insert.

    Architecture notes
    ------------------
    - This class owns no connection state. It receives a db_path and
      opens/closes connections per operation.
    - Thread-safety: a threading.Lock is acquired for every write. The same
      lock pattern used by MemoryManager is used here.
    - The episodes table follows an immutable audit trail: records are never
      deleted. Status transitions (active → superseded, active → retracted)
      are the only mutations.
    - Embeddings are not written here. The EpisodicMemoryReader handles
      embedding population as a separate concern.

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Must be the same file used by MemoryManager.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock    = threading.Lock()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _supersede_existing(
        self,
        conn:         sqlite3.Connection,
        subject:      str,
        episode_type: str,
    ) -> int:
        """
        Mark all active records with (subject, episode_type) as superseded.

        Returns the number of rows updated.
        Called inside an existing write transaction before inserting a new
        active record.
        """
        cursor = conn.execute(
            """
            UPDATE episodes
            SET    status = 'superseded'
            WHERE  subject      = ?
              AND  episode_type = ?
              AND  status       = 'active'
            """,
            (subject, episode_type),
        )
        return cursor.rowcount

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def insert(
        self,
        episode_type:    str,
        subject:         str,
        content:         str,
        source:          str,
        confidence:      float      = 1.0,
        task_id:         str | None = None,
        conversation_id: str | None = None,
        project_context: str | None = "general",
    ) -> int:
        """
        Insert a new active episode.

        If an active record with the same (subject, episode_type) already
        exists, it is marked 'superseded' before the new record is inserted.
        Both records are retained for audit.

        Parameters
        ----------
        episode_type :
            Must be one of VALID_EPISODE_TYPES.
        subject :
            What the episode is about. Used for exact-match retrieval and
            deduplication. Keep concise (< 80 chars recommended).
        content :
            The full text of the episode.
        source :
            Must be one of VALID_SOURCES.
        confidence :
            Float in [0, 1]. Defaults to 1.0 for explicit episodes;
            model-extracted episodes should supply a calibrated value.
        task_id :
            Optional task UUID linking the episode to a specific task session.
        conversation_id :
            Optional conversation UUID for traceability.
        project_context :
            Scope label (e.g. "general", "lora-app-demo"). Defaults to
            "general" so cross-project episodes are naturally grouped.

        Returns
        -------
        int
            The row id (``id`` column) of the newly inserted episode.

        Raises
        ------
        ValueError
            If episode_type or source is not in its respective valid set.
        """
        if episode_type not in VALID_EPISODE_TYPES:
            raise ValueError(
                f"episode_type {episode_type!r} not in VALID_EPISODE_TYPES. "
                f"Valid: {sorted(VALID_EPISODE_TYPES)}"
            )
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source {source!r} not in VALID_SOURCES. "
                f"Valid: {sorted(VALID_SOURCES)}"
            )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence {confidence!r} out of range; must be in [0.0, 1.0]."
            )

        now = time.time()

        with self._lock:
            conn = self._connect()
            try:
                superseded = self._supersede_existing(conn, subject, episode_type)
                if superseded:
                    logger.debug(
                        "insert: superseded %d existing episode(s) for subject=%r type=%r.",
                        superseded, subject, episode_type,
                    )
                cursor = conn.execute(
                    """
                    INSERT INTO episodes
                        (episode_type, subject, content, confidence, source,
                         task_id, conversation_id, project_context,
                         status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (episode_type, subject, content, confidence, source,
                     task_id, conversation_id, project_context, now),
                )
                row_id = cursor.lastrowid
                conn.commit()
                logger.info(
                    "insert: episode id=%d type=%r subject=%r source=%r.",
                    row_id, episode_type, subject, source,
                )
                return row_id
            finally:
                conn.close()

    def retract(
        self,
        subject:      str,
        episode_type: str,
    ) -> int:
        """
        Mark all active episodes with (subject, episode_type) as retracted.

        Unlike supersede — which is an automatic side-effect of insert — retract
        is an explicit operation signalling that the episode is wrong or no
        longer applicable. Retracted records are retained for audit.

        Parameters
        ----------
        subject :
            Exact subject string as stored.
        episode_type :
            Exact episode_type string as stored.

        Returns
        -------
        int
            Number of rows updated (0 if no matching active record exists).
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE episodes
                    SET    status = 'retracted'
                    WHERE  subject      = ?
                      AND  episode_type = ?
                      AND  status       = 'active'
                    """,
                    (subject, episode_type),
                )
                count = cursor.rowcount
                conn.commit()
                logger.info(
                    "retract: retracted %d episode(s) for subject=%r type=%r.",
                    count, subject, episode_type,
                )
                return count
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# EpisodeRecord — return type for all EpisodicMemoryReader retrieval methods
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    id:              int
    episode_type:    str
    subject:         str
    content:         str
    confidence:      float
    source:          str
    task_id:         str | None
    conversation_id: str | None
    project_context: str | None
    status:          str
    created_at:      float
    last_accessed:   float | None


# ---------------------------------------------------------------------------
# EpisodicMemoryReader
# ---------------------------------------------------------------------------

class EpisodicMemoryReader:
    """
    Reads episodes from the `episodes` table in `lora_memory.db`.

    Implements the three retrieval modes defined in §2.6 of
    LOCALIST-Architecture.md:

      Mode 1 — Exact subject match
        By subject string. Returns up to 5 active records ordered by
        confidence DESC, created_at DESC.

      Mode 2 — Type-filtered recency
        High-priority types (preference, correction, decision, workflow)
        for a given project_context. Returns up to 5 active records ordered
        by last_accessed DESC, confidence DESC. Used for session priming.

      Mode 3 — Semantic similarity
        Open-ended query. Cosine ranking over the `embedding` column when
        embeddings are present; falls back to keyword overlap scoring when
        they are absent.

    Side effect — last_accessed update
        Every record returned by any retrieval method has its `last_accessed`
        field updated to the current Unix timestamp. This update is written
        in a single batched UPDATE after the SELECT, inside the same lock.

    Parameters
    ----------
    db_path :
        Path to the SQLite file. Must be the same file used by MemoryManager.
    """

    # Types pulled for session priming (Mode 2).
    _PRIME_TYPES: tuple[str, ...] = (
        "preference", "correction", "decision", "workflow"
    )

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock    = threading.Lock()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _row_to_record(self, row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            id              = row["id"],
            episode_type    = row["episode_type"],
            subject         = row["subject"],
            content         = row["content"],
            confidence      = row["confidence"],
            source          = row["source"],
            task_id         = row["task_id"],
            conversation_id = row["conversation_id"],
            project_context = row["project_context"],
            status          = row["status"],
            created_at      = row["created_at"],
            last_accessed   = row["last_accessed"],
        )

    def _touch_last_accessed(
        self,
        conn: sqlite3.Connection,
        ids:  list[int],
    ) -> float | None:
        """
        Batch-update last_accessed for the given row ids.
        Called inside an existing write transaction.
        Returns the timestamp written, or None when ids is empty.
        """
        if not ids:
            return None
        now = time.time()
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE episodes SET last_accessed = ? WHERE id IN ({placeholders})",
            [now, *ids],
        )
        return now

    # -----------------------------------------------------------------------
    # Public API — three retrieval modes
    # -----------------------------------------------------------------------

    def by_subject(self, subject: str) -> list[EpisodeRecord]:
        """
        Mode 1 — Exact subject match.

        Returns up to 5 active episodes whose subject equals `subject`,
        ordered by confidence DESC then created_at DESC.

        Parameters
        ----------
        subject :
            Exact string to match against the `subject` column.

        Returns
        -------
        list[EpisodeRecord]
            May be empty if no active record matches.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM episodes
                    WHERE  subject = ?
                      AND  status  = 'active'
                    ORDER  BY confidence DESC, created_at DESC
                    LIMIT  5
                    """,
                    (subject,),
                ).fetchall()

                records = [self._row_to_record(r) for r in rows]
                now = self._touch_last_accessed(conn, [r.id for r in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                return records
            finally:
                conn.close()

    def by_recency(
        self,
        project_context: str = "general",
    ) -> list[EpisodeRecord]:
        """
        Mode 2 — Type-filtered recency (session priming).

        Returns up to 5 active episodes of types
        (preference, correction, decision, workflow) for the given
        project_context, ordered by last_accessed DESC then confidence DESC.

        Parameters
        ----------
        project_context :
            Scopes retrieval to a project. Defaults to "general".

        Returns
        -------
        list[EpisodeRecord]
            May be empty.
        """
        placeholders = ",".join("?" * len(self._PRIME_TYPES))

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT * FROM episodes
                    WHERE  episode_type IN ({placeholders})
                      AND  status          = 'active'
                      AND  project_context = ?
                    ORDER  BY last_accessed DESC, confidence DESC
                    LIMIT  5
                    """,
                    (*self._PRIME_TYPES, project_context),
                ).fetchall()

                records = [self._row_to_record(r) for r in rows]
                now = self._touch_last_accessed(conn, [r.id for r in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                return records
            finally:
                conn.close()

    def by_similarity(
        self,
        query:      str,
        top_n:      int   = 5,
        min_score:  float = 0.0,
    ) -> list[EpisodeRecord]:
        """
        Mode 3 — Semantic similarity.

        Scores all active episodes against `query`. If embeddings are
        present on any episode, cosine similarity is used for those rows.
        Rows without embeddings fall back to keyword overlap scoring.
        The combined list is sorted by score DESC and the top `top_n`
        records are returned.

        Note: this class does not hold an embed_fn. Embedding-based scoring
        therefore compares the query's keyword tokens against the episode's
        stored embedding ONLY when a query embedding is available externally.
        In this standalone implementation (no embed_fn injected), all scoring
        falls back to keyword overlap. If embedding support is needed in
        future, subclass or extend with an embed_fn parameter.

        Parameters
        ----------
        query :
            Free-text search string.
        top_n :
            Maximum records to return. Default 5.
        min_score :
            Minimum score threshold. Records below this are excluded.
            Default 0.0 (no filtering).

        Returns
        -------
        list[EpisodeRecord]
            Sorted by relevance score descending. May be empty.
        """
        query_tokens = _tokenize(query)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM episodes
                    WHERE  status = 'active'
                    """
                ).fetchall()

                scored: list[tuple[float, EpisodeRecord]] = []
                for row in rows:
                    record = self._row_to_record(row)
                    combined_text = f"{record.subject} {record.content}"
                    token_set_str = " ".join(_tokenize(combined_text))
                    score = _keyword_score(query_tokens, token_set_str)
                    if score >= min_score:
                        scored.append((score, record))

                scored.sort(key=lambda x: x[0], reverse=True)
                top     = scored[:top_n]
                records = [rec for _, rec in top]
                now = self._touch_last_accessed(conn, [rec.id for rec in records])
                if now is not None:
                    for rec in records:
                        rec.last_accessed = now
                conn.commit()
                logger.debug(
                    "by_similarity: returning %d records for query '%s...'.",
                    len(records), query[:40],
                )
                return records
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Episodic summarization contract  (§2.7 of LOCALIST-Architecture.md)
# ---------------------------------------------------------------------------

# Priority order for bullet ranking. Lower index = higher priority.
# This ordering is load-bearing: it determines which episodes reach the
# model when the 5-bullet cap forces a cut.
_EPISODE_TYPE_PRIORITY: dict[str, int] = {
    "correction":        0,
    "decision":          1,
    "preference":        2,
    "workflow":          3,
    "project_fact":      4,
    "naming_convention": 5,
    "task_completion":   6,
}

# Approximation: one token ≈ 4 characters (conservative for English prose).
# Used to enforce the 20-token-per-bullet ceiling without importing a
# tokeniser. The ceiling is a budget constraint, not a display preference.
_CHARS_PER_TOKEN = 4
_MAX_TOKENS_PER_BULLET = 20
_MAX_BULLET_CHARS = _MAX_TOKENS_PER_BULLET * _CHARS_PER_TOKEN  # 80 chars


def format_episodic_summary(
    episodes:          list["EpisodeRecord"],
    max_bullets:       int   = 5,
    min_confidence:    float = 0.7,
) -> str:
    """
    Format a list of EpisodeRecords into the canonical episodic memory block
    for prompt injection, as specified in §2.7 of LOCALIST-Architecture.md.

    Contract rules enforced
    -----------------------
    1. Only `active` episodes are eligible (callers should pre-filter, but
       this function enforces it defensively).
    2. Only episodes with confidence >= min_confidence (default 0.7) are
       included.
    3. Episodes are sorted by type priority (correction first, task_completion
       last), then by confidence descending as a tiebreaker.
    4. At most `max_bullets` (default 5) bullets are emitted.
    5. Each bullet's content is hard-truncated at 80 characters (≈20 tokens)
       with an ellipsis if truncated. The type annotation and confidence
       score are appended after the content and are NOT counted toward the
       80-char limit.
    6. If no episodes survive filtering, an empty string is returned (no
       label, no placeholder — the caller omits the slot entirely).

    Output format
    -------------
    [EPISODIC MEMORY]
    - {content} ({episode_type}, {confidence:.1f})
    - {content} ({episode_type}, {confidence:.1f})
    ...

    The `[EPISODIC MEMORY]` label and the inline annotation are mandatory.
    The confidence score is formatted to one decimal place.

    Parameters
    ----------
    episodes :
        List of EpisodeRecord objects. Typically the output of one of the
        EpisodicMemoryReader retrieval methods.
    max_bullets :
        Maximum number of bullets to emit. Default 5.
    min_confidence :
        Minimum confidence threshold. Records below this are excluded.
        Default 0.7.

    Returns
    -------
    str
        The formatted block, or "" if no eligible episodes exist.
    """
    # Step 1: filter — active status and confidence threshold
    eligible = [
        ep for ep in episodes
        if ep.status == "active" and ep.confidence >= min_confidence
    ]

    if not eligible:
        return ""

    # Step 2: sort — type priority ASC (lower = higher priority),
    #                confidence DESC as tiebreaker
    eligible.sort(
        key=lambda ep: (
            _EPISODE_TYPE_PRIORITY.get(ep.episode_type, 99),
            -ep.confidence,
        )
    )

    # Step 3: cap at max_bullets
    top = eligible[:max_bullets]

    # Step 4: format each bullet
    lines = ["[EPISODIC MEMORY]"]
    for ep in top:
        content = ep.content
        if len(content) > _MAX_BULLET_CHARS:
            content = content[: _MAX_BULLET_CHARS - 1] + "…"
        lines.append(
            f"- {content} ({ep.episode_type}, {ep.confidence:.1f})"
        )

    return "\n".join(lines)
