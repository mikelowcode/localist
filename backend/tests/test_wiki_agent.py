"""
Tests for wiki_agent — _validate_links() and run() wiring.

Covers:
  1. Link to an existing page resolves silently.
  2. Link to a self-proposed page resolves silently.
  3. Link to neither is flagged; content is provably unchanged.
  4. Case-mismatch resolves after normalization.
  5. Word-count mismatch does NOT resolve (normalization is narrow).
  6. Section scoping — links outside Mapped/Related Pages are ignored.
  7. Missing Mapped/Related Pages sections don't error.
  8. End-to-end run() wiring — unresolved_links in output + warning logged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wiki_agent import (
    Actions,
    CreatePage,
    WikiAgent,
    _validate_links,
    build_user_prompt,
    build_slim_prompt,
    parse_model_xml,
)
from controller_agent import SubTask, TaskStatus


# ---------------------------------------------------------------------------
# Runtime fake — established convention for this file
#
# Use _FakeRuntime (not MagicMock) for any test that exercises WikiAgent.run().
# MagicMock auto-creates infer_with_file as an attribute, causing hasattr()
# in run() to return True and silently routing the test down the
# infer_with_file / build_slim_prompt path instead of the infer() string-prompt
# path. _FakeRuntime has exactly the two methods in the RuntimeClient Protocol
# and deliberately omits infer_with_file, so the routing is deterministic.
# ---------------------------------------------------------------------------

class _FakeRuntime:
    """Protocol-shaped fake — has only infer() and embed(), matching
    RuntimeClient exactly. Deliberately has no infer_with_file, so
    run()'s hasattr() check correctly routes to the infer() string-prompt
    path, the same way a real OMLXRuntimeClient/FoundryRuntimeClient
    without infer_with_file support would."""

    def __init__(self, response: str) -> None:
        self._response = response

    def infer(self, *args, **kwargs) -> str:
        return self._response

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actions(*pages: tuple[str, str]) -> Actions:
    """Create an Actions object with new_pages from (page_name, content) pairs."""
    return Actions(
        new_pages=[
            CreatePage(page_name=name, page_type="RESEARCH_NOTE", content=content)
            for name, content in pages
        ],
        diffs=[],
    )


def _page_with_links(mapped: str = "", related: str = "", concepts: str = "") -> str:
    """Build a page body with optional links in each section."""
    mapped_body = f"- [[{mapped}]] — a mapped page.\n" if mapped else "- null\n"
    related_body = f"- [[{related}]]\n" if related else ""
    concepts_body = f"- Some concept with [[{concepts}]].\n" if concepts else "- A concept.\n"
    return (
        "## Summary\n\nA research note.\n\n"
        "## Details\n\n"
        "### Extracted Concepts\n\n"
        f"{concepts_body}\n"
        "### Mapped Pages\n\n"
        f"{mapped_body}\n"
        "### Proposed New Pages\n\n"
        "- null\n\n"
        "## Related Pages\n\n"
        f"{related_body}\n"
        "## Revision History\n\n"
        "- 2026-06-19 — Created.\n"
    )


# ---------------------------------------------------------------------------
# Test 1 — Link to an existing page resolves silently
# ---------------------------------------------------------------------------

def test_existing_page_resolves():
    actions = _make_actions(
        ("my-page", _page_with_links(mapped="existing-page", related="existing-page"))
    )
    wiki_pages = {"existing-page": "Some content."}
    result = _validate_links(actions, wiki_pages)
    assert result == {}


# ---------------------------------------------------------------------------
# Test 2 — Link to a self-proposed page resolves silently
# ---------------------------------------------------------------------------

def test_self_proposed_page_resolves():
    content_a = _page_with_links(related="new-page-b")
    actions = _make_actions(
        ("new-page-a", content_a),
        ("new-page-b", _page_with_links()),
    )
    wiki_pages = {}  # nothing exists yet
    result = _validate_links(actions, wiki_pages)
    assert result == {}


# ---------------------------------------------------------------------------
# Test 3 — Nonexistent link flagged; content byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_nonexistent_link_flagged_content_unchanged():
    content_before = _page_with_links(related="totally-nonexistent-page")
    actions = _make_actions(("my-page", content_before))
    wiki_pages = {}

    result = _validate_links(actions, wiki_pages)

    assert result == {"my-page": ["totally-nonexistent-page"]}
    # Content must be character-for-character identical after the call
    assert actions.new_pages[0].content == content_before


# ---------------------------------------------------------------------------
# Test 4 — Case-mismatch resolves post-normalization
# ---------------------------------------------------------------------------

def test_case_mismatch_resolves_after_normalization():
    # Real corpus link: [[Localist Master Project Outline]]
    # Real page stem:   localist-master-project-outline
    content = _page_with_links(related="Localist Master Project Outline")
    actions = _make_actions(("my-page", content))
    wiki_pages = {"localist-master-project-outline": "Some content."}

    result = _validate_links(actions, wiki_pages)

    assert result == {}  # resolves — no unresolved entries


# ---------------------------------------------------------------------------
# Test 5 — Word-count mismatch does NOT resolve (normalization is narrow)
# ---------------------------------------------------------------------------

def test_word_count_mismatch_not_resolved():
    # Real corpus defect: [[Localist Software Stack Overview]]
    # vs page stem:       localist-software-stack  (one word missing)
    content = _page_with_links(related="Localist Software Stack Overview")
    actions = _make_actions(("my-page", content))
    wiki_pages = {"localist-software-stack": "Some content."}

    result = _validate_links(actions, wiki_pages)

    assert "my-page" in result
    assert "localist-software-stack-overview" in result["my-page"]


# ---------------------------------------------------------------------------
# Test 6 — Links outside Mapped/Related Pages sections are ignored
# ---------------------------------------------------------------------------

def test_links_outside_sections_ignored():
    # Link appears ONLY in Extracted Concepts, not in Mapped Pages or Related Pages
    content = _page_with_links(concepts="out-of-scope-link")
    actions = _make_actions(("my-page", content))
    wiki_pages = {}  # out-of-scope-link doesn't exist

    result = _validate_links(actions, wiki_pages)

    assert result == {}  # not scanned → not flagged


# ---------------------------------------------------------------------------
# Test 7 — Missing Mapped/Related Pages sections don't error
# ---------------------------------------------------------------------------

def test_missing_sections_no_error():
    # Minimal page with no Mapped Pages or Related Pages headings at all
    content = "## Summary\n\nJust a summary, no section structure.\n"
    actions = _make_actions(("bare-page", content))
    wiki_pages = {}

    result = _validate_links(actions, wiki_pages)

    assert result == {}


# ---------------------------------------------------------------------------
# Test 8 — End-to-end run() wiring
# ---------------------------------------------------------------------------

_RUN_XML = """\
<actions>
  <action name="create_page">
    <page_name>run-test-page</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
## Summary

A test page.

## Details

### Extracted Concepts

- A concept.

### Mapped Pages

- [[existing-wiki-page]] — Exists.

### Proposed New Pages

- null

## Related Pages

- [[existing-wiki-page]]
- [[totally-missing-page]]

## Revision History

- 2026-06-19 — Created.
    </content>
  </action>
</actions>
"""


# ---------------------------------------------------------------------------
# Stub fixtures for prompt-content tests (Rules 1–7)
# ---------------------------------------------------------------------------

_PROMPT_STUBS = dict(
    schema_text  = "# Schema\n",
    templates    = {},
    wiki_context = "(no existing wiki pages)",
    raw_filename = "test.md",
)


# ---------------------------------------------------------------------------
# Test — Rule 7 present in build_user_prompt() output
# ---------------------------------------------------------------------------

def test_build_user_prompt_contains_rule7():
    """build_user_prompt() must contain Rule 7 (verbatim-link-target constraint)."""
    out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    assert "Every [[...]] link target" in out
    assert "MUST exactly match an existing page name" in out
    assert "propose it as a new page" in out


# ---------------------------------------------------------------------------
# Test — Rule 7 present in build_slim_prompt() output, identical wording
# ---------------------------------------------------------------------------

def test_build_slim_prompt_contains_rule7_identical_to_user_prompt():
    """build_slim_prompt() must contain Rule 7, byte-identical to build_user_prompt()."""
    user_out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    slim_out = build_slim_prompt(**_PROMPT_STUBS)

    assert "Every [[...]] link target" in slim_out
    assert "MUST exactly match an existing page name" in slim_out
    assert "propose it as a new page" in slim_out

    # Extract just the Rule-7 text from each and confirm they match exactly.
    import re
    def _extract_rule7(text: str) -> str:
        m = re.search(r"7\. Every \[\[\.\.\..*?instead of linking to a guessed name\.", text, re.DOTALL)
        assert m, f"Rule 7 not found in prompt output"
        return m.group(0)

    assert _extract_rule7(user_out) == _extract_rule7(slim_out), (
        "Rule 7 wording differs between build_user_prompt() and build_slim_prompt()"
    )


# ---------------------------------------------------------------------------
# Test — Rules 1–6 text unchanged in both prompt functions
# ---------------------------------------------------------------------------

_RULE_SUBSTRINGS = [
    "1. Create exactly one RESEARCH_NOTE",
    "2. Details MUST contain three H3 subsections",
    "3. Optionally propose new CONCEPT or SYSTEM pages",
    "4. For existing wiki pages, propose minimal unified diffs",
    "5. Page names MUST be kebab-case",
    "6. Use ",
    "as the date in all Revision History entries",
]


def test_prompt_rules_1_through_6_unchanged():
    """Rules 1–6 must be present and unchanged in both build_user_prompt() and build_slim_prompt()."""
    user_out = build_user_prompt(**_PROMPT_STUBS, raw_content="Some raw content.")
    slim_out = build_slim_prompt(**_PROMPT_STUBS)
    for expected in _RULE_SUBSTRINGS:
        assert expected in user_out, f"build_user_prompt() missing: {expected!r}"
        assert expected in slim_out, f"build_slim_prompt() missing: {expected!r}"


# ---------------------------------------------------------------------------
# Test 8 — End-to-end run() wiring
# ---------------------------------------------------------------------------

def test_run_unresolved_links_in_output_and_logged(tmp_path: Path, caplog):
    # Set up a minimal wiki directory structure
    wiki_dir      = tmp_path / "wiki"
    wiki_dir.mkdir()
    schema_path   = tmp_path / "SCHEMA.md"
    schema_path.write_text("# Schema\n", encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    raw_path      = tmp_path / "raw-input.md"
    raw_path.write_text("Some raw content.\n", encoding="utf-8")

    # Pre-create the existing page so it appears in wiki_pages
    (wiki_dir / "existing-wiki-page.md").write_text("Existing page content.\n", encoding="utf-8")

    rt = _FakeRuntime(_RUN_XML)
    # _FakeRuntime has no infer_with_file → hasattr() returns False → infer() path taken
    assert not hasattr(rt, "infer_with_file")

    agent = WikiAgent(runtime=rt, project_root=tmp_path)

    subtask = MagicMock()
    subtask.subtask_id  = "run-test-0"
    subtask.instruction = "ingest raw-input.md"
    subtask.context = {
        "raw_path":     str(raw_path),
        "wiki_dir":     str(wiki_dir),
        "schema_path":  str(schema_path),
        "templates_dir": str(templates_dir),
        "auto_apply":   False,
    }

    with caplog.at_level(logging.WARNING, logger="wiki_agent"):
        result = agent.run(subtask)

    assert result.status == TaskStatus.COMPLETE
    assert "unresolved_links" in result.output

    unresolved = result.output["unresolved_links"]
    assert "run-test-page" in unresolved
    assert "totally-missing-page" in unresolved["run-test-page"]
    assert "existing-wiki-page" not in unresolved.get("run-test-page", [])

    # Confirm the warning was logged
    assert any(
        "totally-missing-page" in record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
    )


# ---------------------------------------------------------------------------
# Tests — parse_model_xml() strips leading/trailing whitespace from content
# (Open Item 6 write-time fix)
# ---------------------------------------------------------------------------

# Standard model output: Gemma places a newline immediately after <content>.
# This exercises the __CONTENT_N__ placeholder path (all content blocks go
# through _shield_content_blocks, so raw_content is always a placeholder for
# non-empty content; the if-branch in parse_model_xml resolves it back to
# contents[idx]).  The strip() is applied to raw_content after that resolution.
_XML_LEADING_NEWLINE = """\
<actions>
  <action name="create_page">
    <page_name>my-research-note</page_name>
    <page_type>RESEARCH_NOTE</page_type>
    <content>
---
title: My Research Note
type: research-note
---

## Summary

Body text.
    </content>
  </action>
</actions>"""


def test_parse_model_xml_strips_leading_newline_from_content():
    """parse_model_xml must strip the leading newline Gemma places after <content>."""
    actions = parse_model_xml(_XML_LEADING_NEWLINE)
    assert len(actions.new_pages) == 1
    content = actions.new_pages[0].content
    # After stripping, content must start with the frontmatter fence, not a blank line
    assert not content.startswith("\n"), "Leading newline was not stripped"
    assert content.startswith("---"), "Frontmatter fence must be first character"


def test_parse_model_xml_strips_trailing_whitespace_from_content():
    """parse_model_xml must strip trailing whitespace/newlines from content."""
    actions = parse_model_xml(_XML_LEADING_NEWLINE)
    content = actions.new_pages[0].content
    # The XML above has trailing spaces + newline before </content>
    assert not content.endswith("    "), "Trailing indent was not stripped"
    assert content.endswith("Body text."), "Content should end at last real text line"


# XML where content has only trailing whitespace (no leading newline) —
# confirms strip() is applied regardless of which side has the whitespace.
_XML_TRAILING_ONLY = """\
<actions>
  <action name="create_page">
    <page_name>concept-page</page_name>
    <page_type>CONCEPT</page_type>
    <content>## A Concept

Some concept text.
  </content>
  </action>
</actions>"""


def test_parse_model_xml_strips_trailing_only_whitespace():
    """parse_model_xml must strip trailing whitespace even when there is no leading newline."""
    actions = parse_model_xml(_XML_TRAILING_ONLY)
    assert len(actions.new_pages) == 1
    content = actions.new_pages[0].content
    assert not content.endswith("  "), "Trailing spaces were not stripped"
    assert content.startswith("## A Concept")
