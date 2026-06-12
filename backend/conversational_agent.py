"""
LORA — ConversationalAgent (Primary RAG Engine)
================================================
The single reasoning path for all non-ingest user queries.

Architecture
------------
ConversationalAgent is the primary query engine for LORA.  Every query that
is not a wiki ingest flows here.  The pipeline is:

  1. Query MemoryManager — semantic retrieval over the full corpus
     (wiki pages + raw docs, cosine re-ranked when embeddings available)
  2. Select top-k results and inject into the system prompt as grounded context
  3. Single inference call → answer

This replaces the multi-step ResearchAgent pipeline entirely.  The model
(gemma-4-e4b-it-4bit) is strong enough to reason over retrieved wiki context
in a single call.  The structured wiki pages produced by WikiAgent provide
clean, well-organised source material.

Design principles
-----------------
- Single inference call.  No loops, no sub-queries, no synthesis step.
- Corpus-first.  Every query hits MemoryManager before the model.
- Graceful degradation.  If MemoryManager is absent or corpus is empty,
  the model answers from its own knowledge — no crash, no error.
- Wiki-first context.  query_corpus() searches both wiki and raw doc_types;
  wiki pages are preferred because they are structured and agent-verified.
- Source transparency.  AgentResult.output["sources"] lists every document
  path that contributed to the answer.

AgentInterface compliance
--------------------------
    name        → "conversational_agent"
    can_handle  → True for all non-ingest instructions
    run(subtask)→ AgentResult with output["answer"], output["sources"],
                  output["grounded"]

SubTask.context keys (all optional)
------------------------------------
    max_tokens     : int   — model max tokens.              Default 1024.
    temperature    : float — sampling temperature.          Default 0.3.
    max_results    : int   — total corpus docs to inject.   Default 4.
    wiki_threshold : int   — min wiki hits before raw       Default 2.
                             fallback is skipped entirely.
    system         : str   — override system prompt.        Default: _DEFAULT_SYSTEM.

AgentResult.output schema
--------------------------
    answer     : str       — the model's response
    sources    : list[str] — document paths used as context
    grounded   : bool      — True when corpus context was injected

query_corpus() contract (memory_manager.py)
--------------------------------------------
Returns list[DocumentResult] with __slots__:
    .name            str
    .path            Path
    .doc_type        str   — "wiki" or "raw"
    .content         str
    .relevance_score float
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# Instructions containing these keywords are ingest operations — they belong
# to WikiAgent.  can_handle() returns False for these so the Planner routes
# them to wiki_agent instead.
_INGEST_KEYWORDS: frozenset[str] = frozenset({
    "ingest", "update wiki", "create page", "apply diff",
})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = """\
You are LORA, a local research assistant with access to a structured wiki corpus.

Your job is to answer the user's question accurately. Follow these rules:

- Answer directly. Do not open with "Based on the context..." or similar.
- Use the wiki context ONLY if it directly addresses the question being asked.
  If the wiki context is about a different topic, ignore it entirely and answer
  from your own knowledge.
- When using the wiki, cite pages inline (e.g. "According to the Build Order page, …").
- If the question is general knowledge unrelated to the LORA project, answer
  from your own knowledge without referencing the wiki at all.
- Never fabricate project-specific details not present in the context.
- Be concise. Aim for 100–300 words unless the question clearly requires more.
- Plain prose only. No JSON, no XML, no code blocks unless explicitly requested.
"""

# Injected before the user question when corpus results are available.
_CONTEXT_BLOCK = """\
## Wiki Context

{entries}
---

"""

_CONTEXT_ENTRY = """\
### {title}  ({doc_type})

{snippet}
"""

# Characters of each document's content to include in the prompt.
# 2000 chars × 4 docs ≈ 8000 chars of context — comfortable for
# gemma-4-e4b-it-4bit alongside a system prompt.
_MAX_SNIPPET_CHARS = 2000


# ---------------------------------------------------------------------------
# ConversationalAgent
# ---------------------------------------------------------------------------

class ConversationalAgent:
    """
    Corpus-aware single-inference agent.  Primary query engine for LORA.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance (OMLXRuntimeClient or FoundryRuntimeClient).
    memory_manager :
        MemoryManager instance.  When provided, query_corpus() is called
        before every inference to inject grounded wiki context.
        When None, the model answers without corpus context.
    project_root :
        Unused; present for constructor-signature parity with WikiAgent.
    """

    def __init__(
        self,
        runtime:        Any,
        memory_manager: Any | None = None,
        project_root:   Any | None = None,
    ) -> None:
        self._runtime        = runtime
        self._memory_manager = memory_manager
        self._project_root   = project_root  # parity only; not used

    # -----------------------------------------------------------------------
    # AgentInterface
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "conversational_agent"

    def can_handle(self, instruction: str) -> bool:
        """
        Return True for all non-ingest instructions.

        ConversationalAgent is the default handler — it accepts everything
        except explicit wiki ingest requests (which belong to WikiAgent).
        """
        lowered = instruction.lower()
        return not any(kw in lowered for kw in _INGEST_KEYWORDS)

    def run(self, subtask: Any) -> Any:
        """
        RAG pipeline: retrieve → inject → infer → return.

        Steps
        -----
        1. Extract parameters from subtask.context.
        2. Query MemoryManager for top-k relevant documents.
        3. Build context block from retrieved documents.
        4. Assemble final prompt (context + question).
        5. Single runtime.infer() call.
        6. Return AgentResult.
        """
        from controller_agent import AgentResult, TaskStatus

        instruction = subtask.instruction
        context     = subtask.context or {}

        max_tokens     = int(context.get("max_tokens",   1024))
        temperature    = float(context.get("temperature",  0.3))
        max_results    = int(context.get("max_results",   4))
        system         = str(context.get("system", _DEFAULT_SYSTEM))

        logger.info(
            "ConversationalAgent.run() — subtask=%s  chars=%d  max_results=%d",
            subtask.subtask_id, len(instruction), max_results,
        )

        # -- Step 2: Corpus retrieval (wiki-first) ---------------------------
        #
        # Strategy:
        #   1. Query wiki pages only (structured, agent-verified content).
        #   2. If wiki results >= wiki_threshold, use them exclusively —
        #      raw docs would only add duplicate content.
        #   3. If wiki results < wiki_threshold, top up with raw docs to
        #      fill the remaining slots up to max_results.
        #
        # This prevents the duplicate-content problem that occurs when both
        # "lora-master-project-outline.md" (wiki) and
        # "LORA Master Project Outline.md" (raw) are returned for the same
        # query, wasting context window space on identical material.

        context_block = ""
        sources:  list[str] = []
        grounded: bool      = False
        results:  list[Any] = []

        # Minimum wiki hits before raw fallback is skipped entirely.
        wiki_threshold = int(context.get("wiki_threshold", 2))

        if self._memory_manager is not None:
            try:
                # -- Pass 1: wiki pages --
                wiki_results = self._memory_manager.query_corpus(
                    instruction,
                    max_results    = max_results,
                    use_embeddings = True,
                )
                wiki_results = [r for r in wiki_results if r.doc_type == "wiki"]

                if len(wiki_results) >= wiki_threshold:
                    # Wiki corpus is sufficient — use it exclusively.
                    results = wiki_results[:max_results]
                    logger.debug(
                        "Corpus: wiki-only path (%d wiki results, threshold=%d).",
                        len(wiki_results), wiki_threshold,
                    )
                else:
                    # -- Pass 2: top up with raw docs --
                    remaining = max_results - len(wiki_results)
                    raw_results = self._memory_manager.query_corpus(
                        instruction,
                        max_results    = remaining,
                        use_embeddings = True,
                    )
                    raw_results = [r for r in raw_results if r.doc_type == "raw"]
                    results = wiki_results + raw_results[:remaining]
                    logger.debug(
                        "Corpus: wiki+raw fallback path "
                        "(%d wiki, %d raw, threshold=%d).",
                        len(wiki_results), len(raw_results), wiki_threshold,
                    )

                if results:
                    entries = []
                    for doc in results:
                        title   = _path_to_title(str(doc.path))
                        snippet = doc.content[:_MAX_SNIPPET_CHARS]
                        if len(doc.content) > _MAX_SNIPPET_CHARS:
                            snippet += "\n… [truncated]"
                        entries.append(_CONTEXT_ENTRY.format(
                            title    = title,
                            doc_type = doc.doc_type,
                            snippet  = snippet,
                        ))
                        sources.append(str(doc.path))

                    context_block = _CONTEXT_BLOCK.format(entries="\n".join(entries))
                    grounded      = True
                    logger.debug(
                        "Corpus: %d doc(s) injected — %s",
                        len(sources), [os.path.basename(s) for s in sources],
                    )
                else:
                    logger.debug(
                        "Corpus: no results (max_results=%d) "
                        "— answering without context.",
                        max_results,
                    )

            except Exception as exc:
                logger.warning(
                    "ConversationalAgent: corpus query failed (%s) — "
                    "proceeding without context.", exc,
                )

        # -- Step 3: Assemble prompt -----------------------------------------
        if context_block:
            prompt = f"{context_block}## Question\n\n{instruction}"
        else:
            prompt = instruction

        # -- Step 4: Inference -----------------------------------------------
        try:
            answer = self._runtime.infer(
                prompt      = prompt,
                system      = system,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
        except Exception as exc:
            logger.error(
                "ConversationalAgent: inference failed for subtask %s: %s",
                subtask.subtask_id, exc,
            )
            return AgentResult(
                subtask_id = subtask.subtask_id,
                agent_name = self.name,
                status     = TaskStatus.FAILED,
                output     = {},
                error      = f"Inference error: {exc}",
            )

        logger.info(
            "ConversationalAgent.run() complete — answer_chars=%d  "
            "grounded=%s  sources=%d",
            len(answer), grounded, len(sources),
        )

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = {
                "answer":   answer,
                "sources":  sources,
                "grounded": grounded,
            },
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _path_to_title(path: str) -> str:
    """
    Convert a file path to a readable title.

    "wiki/lora-master-project-outline.md" → "Lora Master Project Outline"
    "/abs/path/raw/LORA Build Order.md"   → "LORA Build Order"
    """
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    name = name.replace("-", " ").replace("_", " ")
    words = [w if w.isupper() and len(w) > 1 else w.title() for w in name.split()]
    return " ".join(words)