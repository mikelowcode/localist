"""
Tests for the empty-completion floor in ConversationalAgent's legacy RAG
path (the non-prebuilt path, reached when no controller has pre-assembled a
prompt via PromptBuilder).

Background: Ollama can return a well-formed but zero-content stream
("done": true, no content in between) — confirmed live for queries the
model has no tool grounding for and can't verify from training alone (e.g.
a specific recent/future date). TaskStatus.COMPLETE must never carry an
empty answer.

The prebuilt-prompt path (the one ControllerAgent._execute_plan() actually
uses in production) is deliberately NOT guarded here — that path is only
ever reached via the controller, which owns a smarter forced-web_search
retry before falling back (see
ControllerAgent._dispatch_conversational_with_answer_guard() and its tests
in test_controller_phase4.py::TestEmptyCompletionGuard / TestNonAnswerGuard). Guarding it a
second time here, with a silent substitution, would mask the empty signal
the controller's retry logic depends on. This file covers only the legacy
path, which has no such wrapper and therefore needs its own floor.

Since 2026-11-11 both guards use classify_non_answer(), which also catches
clean-finishing junk replies ("那些\"", " שח") and "Let me search…" stubs —
see TestClassifyNonAnswer below.
"""

import pytest

from unittest.mock import MagicMock

from localist.conversational_agent import (
    ConversationalAgent,
    _EMPTY_RESPONSE_FALLBACK,
    classify_non_answer,
)
from localist.controller_agent import SubTask, TaskStatus


def _make_runtime(infer_return: str = "Normal answer.") -> MagicMock:
    rt = MagicMock()
    rt.infer.return_value = infer_return
    return rt


def _make_subtask(
    instruction: str = "Can you look up something?",
    context: dict | None = None,
) -> SubTask:
    return SubTask(
        subtask_id  = "test-subtask-0",
        agent_name  = "conversational_agent",
        instruction = instruction,
        context     = context or {},
    )


class TestLegacyRagPathEmptyGuard:

    def test_empty_answer_replaced_with_fallback(self):
        rt = _make_runtime(infer_return="")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK
        assert result.output["answer"].strip() != ""
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_whitespace_only_answer_replaced_with_fallback(self):
        """A stream that yields only whitespace chunks is also empty in substance."""
        rt = _make_runtime(infer_return="   \n\t  ")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK

    def test_empty_answer_overrides_grounded_true_from_corpus(self):
        """Guard forces grounded=False even when corpus retrieval would have set it True."""
        rt = _make_runtime(infer_return="")

        doc = MagicMock()
        doc.path = "/wiki/apple-products.md"
        doc.content = "Apple raised MacBook Neo pricing..."
        doc.doc_type = "wiki"
        doc.relevance_score = 0.85

        mm = MagicMock()
        mm.query_corpus.return_value = [doc]

        agent = ConversationalAgent(runtime=rt, memory_manager=mm)
        subtask = _make_subtask()

        result = agent.run(subtask)

        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK
        assert result.output["grounded"] is False
        assert result.output["sources"] == []

    def test_normal_output_passes_through_unmodified(self):
        normal_answer = "2 + 2 equals 4."
        rt = _make_runtime(infer_return=normal_answer)
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(instruction="What is 2 + 2?")

        result = agent.run(subtask)

        assert result.output["answer"] == normal_answer
        assert result.status == TaskStatus.COMPLETE

    def test_non_latin_junk_replaced_with_fallback(self):
        """Real 2026-11-11 OpenRouter reply to an English question."""
        rt = _make_runtime(infer_return=" שח")
        agent = ConversationalAgent(runtime=rt)

        result = agent.run(_make_subtask(instruction="Tell me about the iPhone Air 2 leak"))

        assert result.output["answer"] == _EMPTY_RESPONSE_FALLBACK

    def test_short_answer_not_flagged_on_legacy_path(self):
        """No tool results on this path, so a terse reply is legitimate."""
        rt = _make_runtime(infer_return="Yes.")
        agent = ConversationalAgent(runtime=rt)

        result = agent.run(_make_subtask(instruction="Is Paris in France?"))

        assert result.output["answer"] == "Yes."

    def test_fallback_string_itself_is_never_empty(self):
        """The actual invariant this whole guard exists to protect."""
        assert _EMPTY_RESPONSE_FALLBACK.strip() != ""


class TestPrebuiltPathNotGuardedHere:
    """
    Documents the deliberate asymmetry: the prebuilt-prompt path returns
    whatever the runtime gave it, even if empty — ControllerAgent is the
    only thing that guards this path (see test_controller_phase4.py).
    """

    def test_prebuilt_path_returns_empty_answer_unmodified(self):
        rt = _make_runtime(infer_return="")
        agent = ConversationalAgent(runtime=rt)
        subtask = _make_subtask(context={
            "_prebuilt_prompt": "assembled prompt",
            "_prebuilt_system": "system prompt",
            "_routing": {"fetch_rag": False},
        })

        result = agent.run(subtask)

        assert result.status == TaskStatus.COMPLETE
        assert result.output["answer"] == ""


# Real long junk captured by the Phase 1 live smoke (Nemotron 3 Ultra free).
_LIVE_SOUP_PUNCT = (
    ". |————————))生)))uu)OU)s)))#（)(sThePro))))))))))))))))))-ReReRé.Ré.)))))))))"
    "))))))))))))))))))))-Re-Re-Re-Re-Re-Re-Re-Re)-Re-Re-Re-Re-Re)))))-Re))-Re-Re-Re"
    "-Re-Re-Re)))))-Re))-Re)-Re))-Re)-Re)-Re))-Re)-Re)))-Re)-Re-Re-Re-Re-Re-Re)))))-Re"
)
_LIVE_SOUP_NEWLINES = (
    "E(P\n\n\n\n\n\n(PAN\n\nThe%%%%\n\n\nU(PA\n\n\n\nOO(PAATION\n\nThe\nHH(PAN P"
    "\n\n\n(PA\n\n\n\n\n\n\n\n\n\n\n\n\nA\nA\nA\n\n\nAA\n\nThe\n\n\n\n\n\nA\n\nA"
    "\n\nTheI(PA\tnn\nMM(PA\t\n(Pations\n\n\n\n\n\n\nNnn——\n\n\n\n\nNThe\n\n\nAN(P(P-PNN"
)


class TestClassifyNonAnswer:
    """
    classify_non_answer(): the shared rule set. The first four cases are the
    real replies from 2026-11-11 (chat_turns 1452, 1454, 1467, 1471) plus the
    diagnostic's "The.\n\n\n" — each must be flagged; everything after them
    is a legitimate answer that must not be.
    """

    @pytest.mark.parametrize("answer, had_tools, instruction, expected", [
        # Real failures
        ('那些"', True, 'Tell me more about this news story: "GTA VI…"', "script_mismatch"),
        # Only 17% Greek letters — caught as token soup
        ("i.ζiusfrac.\n\nto.τ", True, 'Tell me more about this news story: "GTA VI…"', "degenerate"),
        # Short reply with stray non-Latin letters, below the soup floor
        ("to.τ", True, "Tell me more", "script_mismatch"),
        # Short soup with enough word-like fragments to pass too_short (live)
        (")V)ide(CA)())Document)Das)", True, "Tell me more about this news story", "degenerate"),
        ("11\n1", True, "Tell me more about this news story", "too_short"),
        # A long answer quoting a Greek letter is fine (covered below);
        # a short one in plain Latin is merely short
        ("It leaked.", True, "Tell me more", "too_short"),
        (" שח", True, 'Tell me more about this news story: "iPhone Air 2…"', "script_mismatch"),
        ("Let me search for this news story to get the latest information.\n"
         "searching for iPhone Air 2 and iPhone 18 dummy models leak GSMArena...",
         True, 'Tell me more about this news story: "iPhone Air 2…"', "intent_stub"),
        ("The.\n\n\n\n\n", True, "Tell me more about this news story", "too_short"),
        ("", True, "anything", "empty"),
        ("  \n\t ", False, "anything", "empty"),
        # Junk script is flagged even without tools
        ('那些"', False, "Hi there", "script_mismatch"),
        # Long token soup, live from Nemotron 3 Ultra free (2026-11-11 smoke)
        (_LIVE_SOUP_PUNCT, True, "Tell me more about this news story", "degenerate"),
        (_LIVE_SOUP_NEWLINES, True, "Tell me more about this news story", "degenerate"),
        (_LIVE_SOUP_PUNCT, False, "Hi there", "degenerate"),
    ])
    def test_flags_non_answers(self, answer, had_tools, instruction, expected):
        assert classify_non_answer(
            answer, had_tool_results=had_tools, instruction=instruction,
        ) == expected

    @pytest.mark.parametrize("answer, had_tools, instruction", [
        # Terse but real, grounded answers (numbers count as words)
        ("Bitcoin is $87,000.", True, "Look up the bitcoin price"),
        ("It's 72°F and sunny in Austin.", True, "Look up Austin weather"),
        ("The answer is 4 (2 + 2).", False, "What is 2+2?"),
        ("Done — saved to notes.md.", False, "Save that to notes.md"),
        ("Paris (France).", False, "Where is the Louvre?"),
        ("It launches November 19, 2026.", True, "When does GTA 6 launch?"),
        # Short replies are fine when no tool grounding was involved
        ("Yes.", False, "Is Paris in France?"),
        ("Hi!", False, "Hello"),
        # A user writing in another script gets an answer in that script
        ("你好，我能帮你什么？", False, "你好"),
        ("东京今天晴，最高气温18度。", True, "东京今天天气怎么样？"),
        # Accented Latin is Latin
        ("Ja, das stimmt. Die Straße ist gesperrt.", True, "Ist die Straße gesperrt?"),
        # A Greek letter inside a real answer is not junk
        ("The circle's area is π r², so about 78.5 square metres.", True, "Look up circle area"),
        # "Let me…" inside a real, long answer is not a stub
        ("Let me summarize what the search found. " + "GTA VI launches on November 19. " * 12,
         True, "Look up GTA 6"),
        # A stub-shaped opener is only a stub when tool results were present
        ("Let me check on that for you.", False, "Can you check something?"),
        # Code scores low on word/whitespace share — fenced code is excluded
        ("Here is the function:\n\n```python\ndef area(r):\n    return 3.14159 * r ** 2\n\n"
         "if __name__ == '__main__':\n    print(area(2))\n```\n\nIt prints 12.566.",
         False, "Write a Python function for circle area"),
        # Markdown tables (like stored turn 1465) are real answers
        ("| Metric | Figure |\n|--------|--------|\n| **Pre-orders** | **7.71 million** |\n"
         "| **Revenue** | **$751 million** |\n| **PS5 share** | **74%** |",
         True, "Look up GTA 6 preorder sales stats."),
    ])
    def test_passes_real_answers(self, answer, had_tools, instruction):
        assert classify_non_answer(
            answer, had_tool_results=had_tools, instruction=instruction,
        ) is None
