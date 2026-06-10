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
from typing import Any, Literal

from pydantic import BaseModel, Field

from controller_agent import (
    AgentResult,
    SubTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV-cache guard rails (match original constants)
# ---------------------------------------------------------------------------

_MAX_WIKI_CHARS   = 6_000   # hard cap on total wiki text injected per call
_RELEVANT_PAGES_N = 4       # max pages selected by keyword relevance
_SUMMARY_LINES    = 3       # lines used for the compact index entry
_MAX_SCHEMA_LINES = 120     # schema is truncated to this many lines


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Taken verbatim from agent_wiki_loop_streaming.py.

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
    page_name: str                                              # kebab-case enforced by prompt
    page_type: Literal["SYSTEM", "CONCEPT", "RESEARCH_NOTE"]   # matches original Literal
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
    """
    One append-only record in the agent journal file.

    Matches the original model's fields exactly: step, timestamp, actions,
    approved, error.  The journal_path is now supplied via SubTask.context
    rather than written to a CWD-relative file.
    """
    step:      str
    timestamp: datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    actions:   Actions | None = None
    approved:  bool    | None = None
    error:     str     | None = None


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file and return its contents as a string."""
    return path.read_text(encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    """Write a UTF-8 string to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def is_text_file(path: Path) -> bool:
    """
    Return True if the file is readable UTF-8 text with no null bytes.

    Matches the original standalone script's detection strategy:
    read all bytes, reject on null byte, reject on UnicodeDecodeError.
    """
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
    """Cheap word-level token set for keyword-overlap relevance scoring."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _one_line_summary(content: str, n: int = _SUMMARY_LINES) -> str:
    """First n non-blank lines joined as a single pipe-separated string."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    return " | ".join(lines[:n])


def build_wiki_context(wiki_pages: dict[str, str], raw_content: str) -> str:
    """
    Build a prompt-safe wiki context block.

    Strategy (matches original agent_wiki_loop_streaming.py exactly):
    - Compact one-line index of ALL pages (always included, token-cheap).
    - Full content for the top RELEVANT_PAGES_N keyword-overlap matches,
      hard-capped at MAX_WIKI_CHARS across all full-content blocks.

    This keeps the KV cache stable regardless of wiki size.
    """
    if not wiki_pages:
        return "(no existing wiki pages)"

    raw_tokens = _tokenize(raw_content)

    # Score every page by keyword overlap with the raw document.
    scored = sorted(
        (
            (len(raw_tokens & _tokenize(content)), name, content)
            for name, content in wiki_pages.items()
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    # Compact index — one line per page, all pages.
    index_lines = ["## WIKI PAGE INDEX (all pages)\n"]
    for _, name, content in scored:
        index_lines.append(f"- {name}: {_one_line_summary(content)}")
    index_block = "\n".join(index_lines)

    # Full content for top-N pages, within the character budget.
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

# Fully-populated RESEARCH_NOTE example embedded in the user prompt.
# Matches the original standalone script verbatim — governs output structure
# for small models (Phi-4-mini) that need a concrete exemplar in the
# highest-attention zone of the prompt.
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
    """
    Assemble the full user-turn prompt for the wiki ingestion model call.

    Matches the original standalone script's prompt structure exactly:
    - Schema (truncated to 120 lines to protect the context budget)
    - Page templates
    - Existing wiki pages (relevance-filtered by build_wiki_context)
    - Raw file to ingest
    - Numbered task rules including today's date for Revision History entries
    - Fully-populated RESEARCH_NOTE example (highest-attention zone)
    - Output rules

    Parameters
    ----------
    schema_text:
        The contents of SCHEMA.md.
    templates:
        Dict of template name → content (keys: SYSTEM, CONCEPT, RESEARCH_NOTE).
    wiki_context:
        Output of build_wiki_context() — relevance-ranked wiki index + full pages.
    raw_filename:
        Basename of the raw file being ingested (for model context only).
    raw_content:
        Full text of the raw document.  Not truncated here — build_wiki_context
        applies the budget cap on the wiki side so the raw document arrives
        intact for accurate extraction.
    """
    template_block = "".join(
        f"### TEMPLATE: {key}\n\n{content}\n"
        for key, content in templates.items()
    )

    # Truncate schema to 120 lines — forces the most load-bearing rules to
    # the top of SCHEMA.md and keeps the context budget predictable.
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
    Assemble the slim user-turn prompt for oMLX native file ingestion.

    Used when the runtime exposes ``infer_with_file()``.  The raw document
    is delivered as a ``type="file"`` content block and processed by
    MarkItDown server-side, so this prompt omits the RAW FILE TO INGEST
    section entirely.

    Keeps everything else from ``build_user_prompt()``:
    - Schema (truncated to 120 lines)
    - Page templates
    - Existing wiki pages (relevance-filtered wiki_context)
    - Task rules and output contract
    - Full RESEARCH_NOTE example

    Parameters
    ----------
    schema_text:
        The contents of SCHEMA.md.
    templates:
        Dict of template name → content (keys: SYSTEM, CONCEPT, RESEARCH_NOTE).
    wiki_context:
        Output of build_wiki_context() — relevance-ranked wiki index + full pages.
        build_wiki_context() is still called by the agent regardless of which
        inference path is taken so the wiki substrate informs the model.
    raw_filename:
        Basename of the raw file being ingested.  Used only in the task
        rules so the model can reference it in Revision History entries.
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
    Extract the first complete <actions>…</actions> block from the raw string.

    Uses str.find / str.rfind rather than regex so that nested angle brackets
    inside <content> blocks do not confuse the extractor.

    Validation is intentionally deferred to parse_model_xml() which calls
    ET.fromstring() on the shielded block.  Pre-validating here caused false
    negatives when <content> blocks contained Markdown syntax (**, [[...]],
    ---) that is valid inside XML text content but trips ET.fromstring() when
    the overall document structure is intact.
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
    XML parsing.

    <content> blocks written by the model contain Markdown syntax such as
    **bold**, [[wiki links]], and --- front-matter separators.  These are
    valid text content in XML but ET.fromstring() rejects them as invalid
    tokens when they appear unescaped.

    Strategy: extract every <content>…</content> block, store the raw text,
    substitute a plain-text placeholder, parse the clean XML structure, then
    restore the original content strings at the call site.

    Returns
    -------
    shielded : str
        The XML string with content replaced by __CONTENT_N__ placeholders.
    contents : list[str]
        The original content strings in insertion order.
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

    The original standalone script — and therefore this module — uses an
    attribute-keyed action schema:

        <actions>
          <action name="create_page">
            <page_name>…</page_name>
            <page_type>…</page_type>
            <content>…</content>
          </action>
          <action name="apply_diff">
            <page_name>…</page_name>
            <diff>…</diff>
          </action>
        </actions>

    <content> blocks may contain Markdown syntax that is invalid XML.
    _shield_content_blocks() extracts them before parsing and they are
    restored after ET.fromstring() succeeds on the clean structure.

    Malformed individual <action> elements (missing required child elements)
    are logged as warnings and skipped.  ValueError is raised only if no
    valid <actions> block exists or the block cannot be parsed as XML at all.

    Raises
    ------
    ValueError
        If no <actions>…</actions> block is found, or if the block fails
        XML parsing entirely.
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
    ``difflib.restore(diff_lines, 2)``, which is designed for *ndiff* format
    (2-char prefixes "  ", "- ", "+ ") — not unified diff format (1-char
    prefixes " ", "-", "+").  Since the prompt instructs the model to emit
    unified diffs, restore() always returned an empty list in practice.

    This implementation correctly parses unified diff @@ hunks and applies
    them with context-line verification, raising ValueError if any hunk
    does not match the file content.

    Hunks are applied in reverse order so that earlier line-number shifts do
    not invalidate the offsets of later hunks.

    Raises
    ------
    ValueError
        If the diff contains no @@ hunks, or if a hunk's context lines do
        not match the actual file content at the expected position.
    """
    hunks = _parse_unified_hunks(diff_text)
    if not hunks:
        raise ValueError("Diff contains no recognisable @@ hunks.")

    result_lines = original.splitlines(keepends=True)
    offset = 0  # cumulative delta from previously applied hunks

    for orig_start, hunk_lines in hunks:
        # orig_start is 1-based; convert to 0-based index and adjust for
        # any previously applied hunks that shifted line numbers.
        idx = orig_start - 1 + offset

        # Reconstruct what the file must contain (context + removed lines)
        # and what it should become (context + added lines).
        before: list[str] = []
        after:  list[str] = []

        for line in hunk_lines:
            if line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            else:
                # Context line (space prefix or bare)
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
    """
    Parse a unified diff string into a list of (orig_start_line, hunk_lines).

    Skips --- / +++ header lines.  Returns an empty list if no @@ markers
    are found.  orig_start_line is the 1-based line number from the @@ header.
    """
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

    Reads a raw document, calls the model through the injected RuntimeClient,
    parses the model's XML response into structured wiki actions, and either
    returns those actions for review or writes them to disk immediately.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance.  Used for all model inference calls.
    project_root :
        Fallback root for resolving wiki_dir, schema_path, and templates_dir
        when they are not supplied in SubTask.context.  Defaults to the
        directory containing this file.
    """

    def __init__(
        self,
        runtime:      Any,          # RuntimeClient — typed Any to avoid circular import
        project_root: Path | None = None,
    ) -> None:
        self._runtime      = runtime
        self._project_root = project_root or Path(__file__).resolve().parent

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
        """
        Return True when the instruction mentions wiki ingestion concepts.
        The Planner uses this as a routing hint for fallback decisions only —
        the primary routing path is the model-generated plan.
        """
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
        8. Return an AgentResult.

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
        # raw_content is always loaded — build_wiki_context() needs it for
        # keyword-overlap scoring regardless of which inference path is taken.

        try:
            schema_text = read_text_file(schema_path)
            templates   = self._load_templates(templates_dir)
            wiki_pages  = self._load_wiki_pages(wiki_dir)
            raw_content = read_text_file(raw_path)
        except Exception as exc:
            return self._fail(subtask, f"File load error: {exc}")

        # -- 4. Build prompt -------------------------------------------------
        #
        # wiki_context is always built — the wiki substrate informs the model
        # regardless of which inference path is taken below.

        wiki_context = build_wiki_context(wiki_pages, raw_content)

        # -- 5. Call model via RuntimeClient ---------------------------------
        #
        # Two paths depending on runtime capability:
        #
        #   infer_with_file() path (oMLX 0.4.2+):
        #     The raw file is sent as a base64 file content block.  oMLX
        #     processes it through MarkItDown server-side before the model
        #     sees it.  The slim prompt omits the RAW FILE TO INGEST section
        #     since the file content arrives via the content block instead.
        #
        #   infer() path (Foundry + all other backends):
        #     The full prompt is built with the raw file content injected as
        #     a string.  Existing behaviour, unchanged.

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

        # -- 9. Build and return AgentResult ---------------------------------

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
        """Extract and validate the raw_path from subtask context."""
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
        """
        Load template files from the templates directory.

        Expected files: system.md, concept.md, research-note.md.
        Missing files are silently skipped — templates are advisory.
        Key mapping matches the original standalone script:
            system.md        → SYSTEM
            concept.md       → CONCEPT
            research-note.md → RESEARCH_NOTE
        """
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

    @staticmethod
    def _apply_changes(
        actions:  Actions,
        wiki_dir: Path,
    ) -> tuple[bool, list[str], list[str], list[str]]:
        """
        Write approved wiki actions to disk.

        Returns
        -------
        applied : bool
            True if at least one change was written without error.
        written : list[str]
            Page names successfully created or patched.
        skipped : list[str]
            Page names skipped (page already existed for new-page actions).
        diff_errors : list[str]
            Page names where diff application failed.
        """
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
        """Append a journal entry to the specified path (silently on failure)."""
        try:
            entry = JournalEntry(step="wiki_agent_run", actions=actions)
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(journal_path, "a", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[wiki_agent] Journal write failed: %s", exc)

    @staticmethod
    def _fail(subtask: SubTask, reason: str) -> AgentResult:
        """Construct a FAILED AgentResult without raising."""
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
    """Verify WikiAgent satisfies AgentInterface at import time (debug aid)."""
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

    # The controller receives this dict from FastAPI.
    result = controller.handle_task({
        "task_id":     str(uuid.uuid4()),
        "instruction": "Ingest the raw file and create a wiki research note.",
        "context": {
            "raw_path":    "/absolute/path/to/your/raw/file.md",
            "wiki_dir":    "/absolute/path/to/your/wiki",
            "schema_path": "/absolute/path/to/your/SCHEMA.md",
            "auto_apply":  False,   # set True to write to disk immediately
        },
    })

    print(json.dumps(result, indent=2))