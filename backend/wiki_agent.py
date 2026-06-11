"""
LORA — WikiAgent
================
Self-contained wiki ingestion agent.  Owns all logic directly.

Previously this module wrapped ``agent_wiki_loop_streaming.py`` and
imported pure functions, Pydantic models, and the SYSTEM_PROMPT from it.
That standalone script has been deleted.  All logic now lives here.

Layer placement
---------------
  ControllerAgent  →  WikiAgent  →  FoundryRuntimeClient (inference)
                                 →  MemoryManager (optional — index updates)

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no stdin, no sys.exit().
- Satisfies the AgentInterface Protocol defined in controller_agent.py.
- All model inference is requested through the injected RuntimeClient —
  never by calling the Foundry HTTP API directly.
- All file I/O is scoped to paths supplied by the Controller via SubTask.context.
  The agent does NOT resolve paths relative to its own __file__ location.
- Changes are written to the wiki directory only when
  SubTask.context["auto_apply"] is True (default: False).  When False,
  the proposed actions are returned in AgentResult.output for the Controller
  or a human review step to approve.
- Journal entries are written to SubTask.context["journal_path"] when
  provided; silently skipped otherwise.
- When a MemoryManager is supplied at construction time, every page
  successfully written to disk is immediately indexed via
  memory_manager.index_document().  This keeps the document index current
  without requiring a full corpus re-scan on the next ResearchAgent call.

SubTask.context schema
----------------------
Required keys
    raw_path : str | Path
        Absolute path to the raw file to ingest (.md or .txt).

Optional keys
    wiki_dir : str | Path
        Wiki pages directory.  Defaults to <project_root>/wiki.
    schema_path : str | Path
        SCHEMA.md path.  Defaults to <project_root>/SCHEMA.md.
    templates_dir : str | Path
        Templates directory.  Defaults to <project_root>/templates.
    journal_path : str | Path
        Path for the agent_journal.jsonl file.  Omit to skip journalling.
    auto_apply : bool
        Write approved changes to disk immediately.  Default False.
    max_tokens : int
        Max tokens for the model call.  Default 2048.
    temperature : float
        Sampling temperature.  Default 0.2.

AgentResult.output schema (on success)
---------------------------------------
    new_pages : list[dict]
        Each dict: {page_name, page_type, content}
    diffs : list[dict]
        Each dict: {page_name, diff}
    applied : bool
        True when changes were written to disk in this call.
    raw_filename : str
        Basename of the ingested raw file.
    wiki_page_count : int
        Number of wiki pages in the index at call time.

    Additional keys present only when auto_apply=True
    written : list[str]
        Page names successfully written or patched.
    skipped : list[str]
        Page names skipped (page already existed for new-page actions).
    diff_errors : list[str]
        Page names where diff application failed.

Porting notes (changes from agent_wiki_loop_streaming.py)
----------------------------------------------------------
The following changes were made during consolidation. The first four are
deliberate architectural removals. The last is a bug fix.

1. sys.exit() removed — all error paths return FAILED AgentResult instead.
2. Interactive prompts removed — file selection and approval loop are gone.
   raw_path comes from SubTask.context; auto_apply replaces the approval prompt.
3. Direct requests.post() removed — all inference goes through RuntimeClient.
4. __file__-relative PROJECT_ROOT removed — project_root is a constructor
   argument with a sensible default; all paths come from SubTask.context.
5. apply_unified_diff fixed — the original used difflib.restore(lines, 2),
   which is designed for ndiff format, not unified diff format.  Since the
   SYSTEM_PROMPT instructs the model to emit unified diffs, restore() always
   returned an empty list.  Replaced with a correct hunk-level parser that
   handles the unified diff format the model actually produces.
"""

from __future__ import annotations

import logging
import re
import textwrap
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from controller_agent import (
    AgentResult,
    SubTask,
    TaskStatus,
)

if TYPE_CHECKING:
    from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-cache guard rails (match original constants)
# ---------------------------------------------------------------------------

_MAX_WIKI_CHARS   = 6_000
_RELEVANT_PAGES_N = 4
_SUMMARY_LINES    = 3
_MAX_SCHEMA_LINES = 120


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a deterministic wiki agent. Your ONLY job is to output a single "
    "valid XML document. You MUST NOT output prose, comments, explanations, or "
    "Markdown fences. Your entire response must be the XML block and nothing else."
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreatePage(BaseModel):
    """A request to create a new wiki page."""
    page_name: str
    page_type: Literal["SYSTEM", "CONCEPT", "RESEARCH_NOTE"]
    content:   str


class ApplyDiff(BaseModel):
    """A request to patch an existing wiki page with a unified diff."""
    page_name: str
    diff:      str


class Actions(BaseModel):
    """The complete set of wiki actions parsed from one model response."""
    new_pages: list[CreatePage] = Field(default_factory=list)
    diffs:     list[ApplyDiff]  = Field(default_factory=list)


class JournalEntry(BaseModel):
    """One append-only record in the agent journal file."""
    step:      str
    timestamp: datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    actions:   Actions | None = None
    approved:  bool    | None = None
    error:     str     | None = None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def is_text_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


# ---------------------------------------------------------------------------
# KV-cache-safe wiki context builder
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _one_line_summary(content: str, n: int = _SUMMARY_LINES) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return " | ".join(lines[:n])


def build_wiki_context(wiki_pages: dict[str, str], raw_content: str) -> str:
    """
    Build a prompt-safe wiki context block.

    Accepts a plain dict (stem → content) as before, so callers that load
    pages from the filesystem or from MemoryManager.get_all_documents() both
    work after normalising to this shape.
    """
    if not wiki_pages:
        return "(no existing wiki pages)"

    raw_tokens = _tokenize(raw_content)

    scored = sorted(
        (
            (len(raw_tokens & _tokenize(content)), name, content)
            for name, content in wiki_pages.items()
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    index_lines = ["## WIKI PAGE INDEX (all pages)\n"]
    for _, name, content in scored:
        index_lines.append(f"- {name}: {_one_line_summary(content)}")
    index_block = "\n".join(index_lines)

    full_blocks: list[str] = []
    budget = _MAX_WIKI_CHARS

    for _, name, content in scored[:_RELEVANT_PAGES_N]:
        chunk = f"---\n## WIKI PAGE (full): {name}\n\n{content}\n"
        if len(chunk) > budget:
            full_blocks.append(
                chunk[:budget] + "\n... [truncated to fit context budget]\n"
            )
            break
        full_blocks.append(chunk)
        budget -= len(chunk)
        if budget <= 0:
            break

    full_block = "\n".join(full_blocks)
    return f"{index_block}\n\n{full_block}".strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_EXAMPLE = """
# EXAMPLE OUTPUT (follow this structure exactly)
<actions>
  <action name="create_page">
    <page_name>example-research-note</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
---
type: RESEARCH_NOTE
status: draft
source: example-raw-file.txt
---

# example-research-note

## Summary

Two to five sentences summarising the raw file. No speculation. Grounded only
in what the raw file explicitly states.

## Details

### Extracted Concepts

- Concept one extracted directly from the raw file.
- Concept two extracted directly from the raw file.

### Mapped Pages

- [[existing-page]] — reason this page is relevant.

### Proposed New Pages

- `new-page-name` (CONCEPT) — reason a new page is justified.

## Related Pages

- [[existing-page]]

## Revision History

- YYYY-MM-DD — Initial research note created from example-raw-file.txt.
    </content>
  </action>
</actions>
"""


def build_user_prompt(
    schema_text:  str,
    templates:    dict[str, str],
    wiki_context: str,
    raw_filename: str,
    raw_content:  str,
) -> str:
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    schema_lines   = schema_text.splitlines()
    schema_snippet = "\n".join(schema_lines[:_MAX_SCHEMA_LINES])
    if len(schema_lines) > _MAX_SCHEMA_LINES:
        schema_snippet += "\n... [schema truncated for context budget]"

    today = date.today().isoformat()

    prompt = f"""\
# SCHEMA (rules you must follow)
{schema_snippet}

# PAGE TEMPLATES
{template_block}

# EXISTING WIKI PAGES
{wiki_context}

# RAW FILE TO INGEST
Filename: {raw_filename}

{textwrap.dedent(raw_content).strip()}

# YOUR TASK
1. Create exactly one RESEARCH_NOTE for the raw file. It MUST include all five
   sections in order: Summary · Details · Related Pages · Revision History,
   each as an H2 heading, plus front-matter at the top.
2. Details MUST contain three H3 subsections: Extracted Concepts · Mapped Pages
   · Proposed New Pages. Use the string "null" when a subsection is empty.
3. Optionally propose new CONCEPT or SYSTEM pages only if clearly justified.
4. For existing wiki pages, propose minimal unified diffs only where necessary.
5. Page names MUST be kebab-case (lowercase letters, digits, hyphens only).
6. Use {today} as the date in all Revision History entries.
{_EXAMPLE}
OUTPUT RULES (read before generating):
- Output ONLY the <actions> XML block. Nothing before it. Nothing after it.
- No prose, no code fences, no explanations outside the XML.
- Every <content> block MUST follow the example structure above exactly.\
"""
    return textwrap.dedent(prompt).strip()


def build_slim_prompt(
    schema_text:  str,
    templates:    dict[str, str],
    wiki_context: str,
    raw_filename: str,
) -> str:
    """
    Slim prompt for the oMLX native file ingestion path.

    Omits the RAW FILE TO INGEST section — the file content is delivered as
    a ``type="file"`` content block and processed by MarkItDown server-side.
    """
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    schema_lines   = schema_text.splitlines()
    schema_snippet = "\n".join(schema_lines[:_MAX_SCHEMA_LINES])
    if len(schema_lines) > _MAX_SCHEMA_LINES:
        schema_snippet += "\n... [schema truncated for context budget]"

    today = date.today().isoformat()

    prompt = f"""\
# SCHEMA (rules you must follow)
{schema_snippet}

# PAGE TEMPLATES
{template_block}

# EXISTING WIKI PAGES
{wiki_context}

# YOUR TASK
The file "{raw_filename}" has been provided directly for processing.
1. Create exactly one RESEARCH_NOTE for the raw file. It MUST include all five
   sections in order: Summary · Details · Related Pages · Revision History,
   each as an H2 heading, plus front-matter at the top.
2. Details MUST contain three H3 subsections: Extracted Concepts · Mapped Pages
   · Proposed New Pages. Use the string "null" when a subsection is empty.
3. Optionally propose new CONCEPT or SYSTEM pages only if clearly justified.
4. For existing wiki pages, propose minimal unified diffs only where necessary.
5. Page names MUST be kebab-case (lowercase letters, digits, hyphens only).
6. Use {today} as the date in all Revision History entries.
{_EXAMPLE}
OUTPUT RULES (read before generating):
- Output ONLY the <actions> XML block. Nothing before it. Nothing after it.
- No prose, no code fences, no explanations outside the XML.
- Every <content> block MUST follow the example structure above exactly.\
"""
    return textwrap.dedent(prompt).strip()


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

def _extract_actions_xml(text: str) -> str | None:
    """
    Extract the first complete <actions>…</actions> block.

    Uses str.find / str.rfind rather than regex so that nested angle brackets
    inside <content> blocks do not confuse the extractor.  Validation is
    deferred to parse_model_xml() via ET.fromstring() on the shielded block.
    """
    start = text.find("<actions")
    if start == -1:
        return None
    end = text.rfind("</actions>")
    if end == -1:
        return None
    return text[start : end + len("</actions>")].strip()


def _shield_content_blocks(xml_text: str) -> tuple[str, list[str]]:
    """
    Replace <content>...</content> blocks with safe placeholders before
    XML parsing to prevent Markdown syntax inside them from tripping ET.
    """
    contents: list[str] = []

    def _replacer(m: re.Match) -> str:
        contents.append(m.group(1))
        return f"<content>__CONTENT_{len(contents) - 1}__</content>"

    shielded = re.sub(
        r"<content>(.*?)</content>",
        _replacer,
        xml_text,
        flags=re.DOTALL,
    )
    return shielded, contents


def parse_model_xml(raw_output: str) -> Actions:
    """
    Parse the model's XML response into an Actions object.

    Raises
    ------
    ValueError
        If no <actions>…</actions> block is found, or if it cannot be parsed.
    """
    xml_block = _extract_actions_xml(raw_output)
    if xml_block is None:
        raise ValueError("No valid <actions> XML block found in model output.")

    shielded, contents = _shield_content_blocks(xml_block)
    try:
        root = ET.fromstring(shielded)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse <actions> XML: {exc}") from exc

    if root.tag != "actions":
        raise ValueError(f"Root XML element must be <actions>, got <{root.tag}>.")

    new_pages: list[dict[str, str]] = []
    diffs:     list[dict[str, str]] = []

    for action in root.findall("action"):
        action_name = action.get("name", "")

        if action_name == "create_page":
            raw_content = action.findtext("content") or ""
            if raw_content.startswith("__CONTENT_") and raw_content.endswith("__"):
                idx = int(raw_content[10:-2])
                raw_content = contents[idx] if idx < len(contents) else ""
            entry = {
                "page_name": (action.findtext("page_name") or "").strip(),
                "page_type": (action.findtext("page_type") or "").strip(),
                "content":   raw_content,
            }
            if not entry["page_name"] or not entry["page_type"]:
                logger.warning(
                    "parse_model_xml: skipping create_page with missing page_name or page_type."
                )
                continue
            new_pages.append(entry)

        elif action_name == "apply_diff":
            entry = {
                "page_name": (action.findtext("page_name") or "").strip(),
                "diff":      action.findtext("diff") or "",
            }
            if not entry["page_name"] or not entry["diff"]:
                logger.warning(
                    "parse_model_xml: skipping apply_diff with missing page_name or diff."
                )
                continue
            diffs.append(entry)

        else:
            logger.debug("parse_model_xml: ignoring unknown action name=%r.", action_name)

    logger.debug(
        "parse_model_xml: parsed %d create_page and %d apply_diff action(s).",
        len(new_pages), len(diffs),
    )
    return Actions.model_validate({"new_pages": new_pages, "diffs": diffs})


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------

def apply_unified_diff(original: str, diff_text: str) -> str:
    """
    Apply a unified diff string to the original file content and return the
    patched result.

    BUG FIX vs original: agent_wiki_loop_streaming.py called
    ``difflib.restore(diff_lines, 2)``, which is designed for *ndiff* format,
    not unified diff format.  This implementation correctly parses @@ hunks.

    Raises
    ------
    ValueError
        If the diff contains no @@ hunks, or if context lines don't match.
    """
    hunks = _parse_unified_hunks(diff_text)
    if not hunks:
        raise ValueError("Diff contains no recognisable @@ hunks.")

    result_lines = original.splitlines(keepends=True)
    offset = 0

    for orig_start, hunk_lines in hunks:
        idx = orig_start - 1 + offset

        before: list[str] = []
        after:  list[str] = []

        for line in hunk_lines:
            if line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            else:
                ctx = line[1:] if line.startswith(" ") else line
                before.append(ctx)
                after.append(ctx)

        actual = result_lines[idx : idx + len(before)]
        if actual != before:
            raise ValueError(
                f"Diff hunk at original line {orig_start} does not match file content.\n"
                f"Expected: {''.join(before)!r}\n"
                f"Got:      {''.join(actual)!r}"
            )

        result_lines[idx : idx + len(before)] = after
        offset += len(after) - len(before)

    return "".join(result_lines)


def _parse_unified_hunks(diff_text: str) -> list[tuple[int, list[str]]]:
    hunks:         list[tuple[int, list[str]]] = []
    current_start: int        = 0
    current_lines: list[str]  = []
    in_hunk = False

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("@@"):
            if in_hunk and current_lines:
                hunks.append((current_start, current_lines))
            m = re.match(r"^@@ -(\d+)", line)
            current_start = int(m.group(1)) if m else 1
            current_lines = []
            in_hunk = True
        elif in_hunk:
            current_lines.append(line)

    if in_hunk and current_lines:
        hunks.append((current_start, current_lines))

    return hunks


# ---------------------------------------------------------------------------
# Routing keywords used by can_handle()
# ---------------------------------------------------------------------------

_WIKI_KEYWORDS: frozenset[str] = frozenset({
    "ingest", "wiki", "research note", "research_note",
    "create page", "update page", "raw file", "document",
    "knowledge base", "summarise", "summarize", "extract concepts",
})


# ---------------------------------------------------------------------------
# WikiAgent
# ---------------------------------------------------------------------------

class WikiAgent:
    """
    Self-contained wiki ingestion agent satisfying the AgentInterface Protocol.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance.  Used for all model inference calls.
    project_root :
        Fallback root for resolving wiki_dir, schema_path, and templates_dir
        when they are not supplied in SubTask.context.  Defaults to the
        directory containing this file.
    memory_manager :
        Optional SQLite-backed MemoryManager.  When supplied, every page
        written to disk is immediately passed to
        memory_manager.index_document() so the document index stays current
        without a full corpus re-scan.  When absent, disk writes are
        unchanged — the index is simply not updated automatically.
    """

    def __init__(
        self,
        runtime:        Any,
        project_root:   Path | None = None,
        memory_manager: "MemoryManager | None" = None,
    ) -> None:
        self._runtime        = runtime
        self._project_root   = project_root or Path(__file__).resolve().parent
        self._memory_manager = memory_manager

    # -----------------------------------------------------------------------
    # AgentInterface — name
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "wiki_agent"

    # -----------------------------------------------------------------------
    # AgentInterface — can_handle
    # -----------------------------------------------------------------------

    def can_handle(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return any(kw in lowered for kw in _WIKI_KEYWORDS)

    # -----------------------------------------------------------------------
    # AgentInterface — run
    # -----------------------------------------------------------------------

    def run(self, subtask: SubTask) -> AgentResult:
        """
        Ingest one raw file and return proposed (or applied) wiki actions.

        Pipeline
        --------
        1. Resolve and validate paths from subtask.context.
        2. Load schema, templates, wiki index, and raw document.
        3. Build wiki_context (always — informs model regardless of path).
        4. Detect runtime capability and call model:
             - infer_with_file() if runtime supports it (oMLX 0.4.2+)
             - infer() with full string prompt otherwise (Foundry + others)
        5. Parse the model's XML response into Actions.
        6. Optionally journal the result.
        7. Optionally write changes to disk (auto_apply=True only).
        8. Index newly written pages in MemoryManager (if available).
        9. Return an AgentResult.

        No stdin.  No sys.exit().  No interactive prompts.
        """
        ctx = subtask.context

        # -- 1. Resolve paths ------------------------------------------------

        try:
            raw_path = self._resolve_raw_path(ctx)
        except (KeyError, ValueError) as exc:
            return self._fail(subtask, f"Context error: {exc}")

        wiki_dir      = Path(ctx.get("wiki_dir",      self._project_root / "wiki"))
        schema_path   = Path(ctx.get("schema_path",   self._project_root / "SCHEMA.md"))
        templates_dir = Path(ctx.get("templates_dir", self._project_root / "templates"))
        journal_path  = Path(ctx["journal_path"]) if "journal_path" in ctx else None
        auto_apply    = bool(ctx.get("auto_apply",  False))
        max_tokens    = int(ctx.get("max_tokens",   2048))
        temperature   = float(ctx.get("temperature", 0.2))

        # -- 2. Validate required paths --------------------------------------

        missing = [
            label for label, p in [
                ("SCHEMA.md",  schema_path),
                ("templates/", templates_dir),
                ("wiki/",      wiki_dir),
                ("raw_path",   raw_path),
            ]
            if not p.exists()
        ]
        if missing:
            return self._fail(subtask, f"Required paths not found: {missing}")

        logger.info(
            "[%s] Ingesting '%s' — auto_apply=%s",
            self.name, raw_path.name, auto_apply,
        )

        # -- 3. Load inputs --------------------------------------------------
        #
        # wiki_pages dict is built either from the MemoryManager index (fast,
        # no filesystem walk) or from disk (fallback when no manager is present).
        # raw_content is always loaded from disk — needed for build_wiki_context()
        # keyword scoring and for the string-prompt infer() path.

        try:
            schema_text = read_text_file(schema_path)
            templates   = self._load_templates(templates_dir)
            raw_content = read_text_file(raw_path)

            if self._memory_manager is not None:
                wiki_pages = self._load_wiki_pages_from_index(wiki_dir)
                logger.debug(
                    "[%s] wiki_pages loaded from MemoryManager index (%d pages).",
                    self.name, len(wiki_pages),
                )
            else:
                wiki_pages = self._load_wiki_pages(wiki_dir)
                logger.debug(
                    "[%s] wiki_pages loaded from filesystem (%d pages).",
                    self.name, len(wiki_pages),
                )
        except Exception as exc:
            return self._fail(subtask, f"File load error: {exc}")

        # -- 4. Build prompt -------------------------------------------------

        wiki_context = build_wiki_context(wiki_pages, raw_content)

        # -- 5. Call model via RuntimeClient ---------------------------------

        use_file_upload = hasattr(self._runtime, "infer_with_file")
        logger.debug(
            "[%s] inference path: %s",
            self.name,
            "infer_with_file (oMLX native)" if use_file_upload else "infer (string prompt)",
        )

        try:
            if use_file_upload:
                slim_prompt = build_slim_prompt(
                    schema_text  = schema_text,
                    templates    = templates,
                    wiki_context = wiki_context,
                    raw_filename = raw_path.name,
                )
                logger.debug(
                    "[%s] slim_prompt_chars=%d  file=%s",
                    self.name, len(slim_prompt), raw_path.name,
                )
                raw_output = self._runtime.infer_with_file(
                    file_path   = raw_path,
                    prompt      = slim_prompt,
                    system      = SYSTEM_PROMPT,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
            else:
                try:
                    user_prompt = build_user_prompt(
                        schema_text  = schema_text,
                        templates    = templates,
                        wiki_context = wiki_context,
                        raw_filename = raw_path.name,
                        raw_content  = raw_content,
                    )
                except Exception as exc:
                    return self._fail(subtask, f"Prompt build error: {exc}")
                logger.debug(
                    "[%s] user_prompt_chars=%d",
                    self.name, len(user_prompt),
                )
                raw_output = self._runtime.infer(
                    system      = SYSTEM_PROMPT,
                    prompt      = user_prompt,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
        except (RuntimeError, ValueError) as exc:
            return self._fail(subtask, f"Inference error: {exc}")

        # -- 6. Parse XML response -------------------------------------------

        try:
            actions = parse_model_xml(raw_output)
        except ValueError as exc:
            return self._fail(subtask, f"XML parse error: {exc}")

        # -- 7. Optionally journal the result --------------------------------

        if journal_path is not None:
            self._write_journal(actions, journal_path)

        # -- 8. Optionally apply changes to disk -----------------------------

        applied    = False
        skipped:   list[str] = []
        written:   list[str] = []
        diff_errs: list[str] = []

        if auto_apply:
            applied, written, skipped, diff_errs = self._apply_changes(
                actions, wiki_dir
            )

            # -- 9. Update MemoryManager index for every written page --------
            #
            # Called here, after disk writes are confirmed, so the index only
            # ever reflects pages that actually exist on disk.  Each call is
            # idempotent (content-hash checked inside index_document), so
            # re-indexing an unchanged page is a no-op.

            if self._memory_manager is not None and written:
                for page_name in written:
                    page_path = wiki_dir / f"{page_name}.md"
                    if page_path.exists():
                        try:
                            self._memory_manager.index_document(
                                path     = page_path,
                                doc_type = "wiki",
                                embed    = True,
                            )
                            logger.debug(
                                "[%s] Indexed '%s' in MemoryManager.", self.name, page_name
                            )
                        except Exception as exc:
                            # Non-fatal — disk write succeeded; index can be
                            # rebuilt by calling index_directory() later.
                            logger.warning(
                                "[%s] MemoryManager.index_document failed for '%s': %s",
                                self.name, page_name, exc,
                            )

        # -- 10. Build and return AgentResult --------------------------------

        output: dict[str, Any] = {
            "new_pages":       [p.model_dump() for p in actions.new_pages],
            "diffs":           [d.model_dump() for d in actions.diffs],
            "applied":         applied,
            "raw_filename":    raw_path.name,
            "wiki_page_count": len(wiki_pages),
        }

        if auto_apply:
            output["written"]     = written
            output["skipped"]     = skipped
            output["diff_errors"] = diff_errs

        logger.info(
            "[%s] Complete — new_pages=%d  diffs=%d  applied=%s",
            self.name,
            len(actions.new_pages),
            len(actions.diffs),
            applied,
        )

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = output,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_raw_path(ctx: dict[str, Any]) -> Path:
        if "raw_path" not in ctx:
            raise KeyError(
                "'raw_path' is required in SubTask.context. "
                "Provide the absolute path to the file to ingest."
            )
        path = Path(ctx["raw_path"])
        if not path.exists():
            raise ValueError(f"raw_path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"raw_path is not a file: {path}")
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(
                f"raw_path must be a .md or .txt file, got: {path.suffix}"
            )
        if not is_text_file(path):
            raise ValueError(
                f"raw_path does not appear to be a UTF-8 text file: {path}"
            )
        return path

    @staticmethod
    def _load_templates(templates_dir: Path) -> dict[str, str]:
        templates: dict[str, str] = {}
        for name in ["system", "concept", "research-note"]:
            p = templates_dir / f"{name}.md"
            if p.exists():
                key = name.upper().replace("-", "_")
                templates[key] = read_text_file(p)
        return templates

    @staticmethod
    def _load_wiki_pages(wiki_dir: Path) -> dict[str, str]:
        """Load all .md files from the wiki directory (stem → content)."""
        if not wiki_dir.exists():
            return {}
        return {
            p.stem: read_text_file(p)
            for p in sorted(wiki_dir.iterdir())
            if p.is_file() and p.suffix == ".md"
        }

    def _load_wiki_pages_from_index(self, wiki_dir: Path) -> dict[str, str]:
        """
        Load wiki page content from the MemoryManager index.

        Falls back to the filesystem walk if the index returns no results
        for this wiki_dir (e.g. on first run before any pages are indexed).
        Filters by the absolute wiki_dir so pages from other projects don't
        bleed in when a single MemoryManager is shared.
        """
        assert self._memory_manager is not None  # guarded by caller

        docs = self._memory_manager.get_all_documents(doc_type="wiki")

        # Filter to pages that live under this wiki_dir
        wiki_dir_str = str(wiki_dir.resolve())
        filtered = {
            d.name: d.content
            for d in docs
            if str(d.path).startswith(wiki_dir_str)
        }

        if not filtered and wiki_dir.exists():
            logger.debug(
                "[%s] MemoryManager index empty for wiki_dir=%s — falling back to filesystem.",
                self.name, wiki_dir,
            )
            return self._load_wiki_pages(wiki_dir)

        return filtered

    @staticmethod
    def _apply_changes(
        actions:  Actions,
        wiki_dir: Path,
    ) -> tuple[bool, list[str], list[str], list[str]]:
        written:     list[str] = []
        skipped:     list[str] = []
        diff_errors: list[str] = []

        for entry in actions.new_pages:
            path = wiki_dir / f"{entry.page_name}.md"
            if path.exists():
                logger.warning("[wiki_agent] Skipping existing page: %s", entry.page_name)
                skipped.append(entry.page_name)
                continue
            write_text_file(path, entry.content)
            logger.info("[wiki_agent] Created: %s", path)
            written.append(entry.page_name)

        for entry in actions.diffs:
            path = wiki_dir / f"{entry.page_name}.md"
            if not path.exists():
                logger.warning("[wiki_agent] Diff target missing: %s", entry.page_name)
                diff_errors.append(entry.page_name)
                continue
            original = read_text_file(path)
            try:
                updated = apply_unified_diff(original, entry.diff)
                write_text_file(path, updated)
                logger.info("[wiki_agent] Patched: %s", path)
                written.append(entry.page_name)
            except Exception as exc:
                logger.error("[wiki_agent] Diff failed for %s: %s", entry.page_name, exc)
                diff_errors.append(entry.page_name)

        applied = bool(written)
        return applied, written, skipped, diff_errors

    @staticmethod
    def _write_journal(actions: Actions, journal_path: Path) -> None:
        try:
            entry = JournalEntry(step="wiki_agent_run", actions=actions)
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(journal_path, "a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[wiki_agent] Journal write failed: %s", exc)

    @staticmethod
    def _fail(subtask: SubTask, reason: str) -> AgentResult:
        logger.error("[wiki_agent] subtask %s failed: %s", subtask.subtask_id, reason)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "wiki_agent",
            status     = TaskStatus.FAILED,
            output     = {},
            error      = reason,
        )


# ---------------------------------------------------------------------------
# Protocol conformance check
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    from controller_agent import AgentInterface

    class _MockRuntime:
        def infer(self, *a, **kw) -> str:  return ""
        def embed(self, text: str) -> list: return []

    agent = WikiAgent(runtime=_MockRuntime())
    assert isinstance(agent, AgentInterface), (
        "WikiAgent does not satisfy the AgentInterface Protocol."
    )
    logger.debug("Protocol conformance check passed.")


# ---------------------------------------------------------------------------
# Wiring example — for reference, not executed in production
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import uuid
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, stream=sys.stdout)

    from foundry_runtime_client import FoundryRuntimeClient
    from controller_agent import ControllerAgent

    runtime    = FoundryRuntimeClient()
    wiki_agent = WikiAgent(runtime=runtime)
    controller = ControllerAgent(runtime=runtime, agents=[wiki_agent])

    result = controller.handle_task({
        "task_id":     str(uuid.uuid4()),
        "instruction": "Ingest the raw file and create a wiki research note.",
        "context": {
            "raw_path":    "/absolute/path/to/your/raw/file.md",
            "wiki_dir":    "/absolute/path/to/your/wiki",
            "schema_path": "/absolute/path/to/your/SCHEMA.md",
            "auto_apply":  False,
        },
    })

    print(json.dumps(result, indent=2))