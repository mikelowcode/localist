"""
LORA — ResearchAgent
=====================
A multi-step, iterative research and synthesis agent.

Layer placement
---------------
  ControllerAgent  →  ResearchAgent  →  FoundryRuntimeClient (inference + embed)
                                     →  wiki/ directory          (read-only)
                                     →  raw/ directory           (read-only)

Architectural contract
----------------------
- Pure Python module.  No FastAPI, no HTTP, no stdin, no sys.exit().
- Satisfies the AgentInterface Protocol defined in controller_agent.py.
- All model inference is requested through the injected RuntimeClient.
- The agent is READ-ONLY with respect to the wiki and raw directories.
  It surfaces findings in AgentResult.output; it never writes wiki pages.
  Writing is the WikiAgent's responsibility.
- No user-facing prompts, no interactive loops.

Responsibilities
----------------
The ResearchAgent handles tasks that require *reasoning across multiple
sources* rather than ingesting a single raw file.  Concretely:

  1. Query decomposition  — break a complex question into focused sub-queries.
  2. Corpus retrieval     — find relevant wiki pages and raw files for each
                           sub-query using keyword overlap and embedding
                           cosine similarity (when embeddings are available).
  3. Iterative reading    — read retrieved documents, extract relevant passages,
                           and decide whether more sources are needed.
  4. Claim extraction     — identify discrete factual claims with source refs.
  5. Synthesis            — combine claims into a structured research report
                           with provenance for every assertion.
  6. Gap detection        — note what the corpus does NOT answer, so the
                           Controller or WikiAgent can act on those gaps.

Difference from WikiAgent
--------------------------
  WikiAgent   : one raw file in  →  structured wiki actions out  (write path)
  ResearchAgent : a question in   →  research report + sources out (read path)

They are complementary halves of a research workflow.  A typical multi-step
task might be:
  1. Controller → ResearchAgent  ("what do we know about X?")
  2. Controller → WikiAgent      ("ingest this new raw file about X")
  3. Controller → ResearchAgent  ("synthesise updated findings about X")

SubTask.context schema
-----------------------
Required keys
    query : str
        The research question or topic to investigate.

Optional keys
    wiki_dir : str | Path
        Wiki pages directory.  Defaults to <project_root>/wiki.
    raw_dir : str | Path
        Raw files directory.  Defaults to <project_root>/raw.
    max_sources : int
        Maximum number of source documents to retrieve per sub-query.
        Default 5.
    max_iterations : int
        Maximum read-and-reflect iterations before forcing synthesis.
        Default 3.
    max_tokens_per_call : int
        Token budget per individual model call.  Default 1024.
    temperature : float
        Sampling temperature.  Default 0.2.
    use_embeddings : bool
        If True, supplement keyword retrieval with embedding cosine
        similarity re-ranking (requires embed() support in the runtime).
        Default False — safe to enable once your embedding model is
        confirmed working via health_check().

AgentResult.output schema (on success)
---------------------------------------
    report : str
        The synthesised research report in plain Markdown.
    claims : list[dict]
        Each dict: {claim: str, source: str, confidence: "high"|"medium"|"low"}
    sources : list[dict]
        Each dict: {name: str, path: str, type: "wiki"|"raw", relevance_score: float}
    sub_queries : list[str]
        The decomposed sub-queries that were investigated.
    gaps : list[str]
        Topics the corpus did not adequately cover.
    iterations : int
        Number of read-and-reflect iterations actually performed.
    query : str
        Echo of the original query for downstream reference.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from controller_agent import (
    AgentResult,
    SubTask,
    TaskStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing keywords for can_handle()
# ---------------------------------------------------------------------------

_RESEARCH_KEYWORDS: frozenset[str] = frozenset({
    "research", "investigate", "analyse", "analyze", "synthesise", "synthesize",
    "what do we know", "find information", "look into", "explore", "review",
    "compare", "contrast", "summarise findings", "summarize findings",
    "what is known", "deep dive", "report on", "study",
})


# ---------------------------------------------------------------------------
# Lightweight text utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Word-level token set for keyword overlap scoring."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_score(query_tokens: set[str], document: str) -> float:
    """Jaccard-like overlap between query tokens and document tokens."""
    doc_tokens = _tokenize(document)
    if not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _truncate(text: str, max_chars: int) -> str:
    """Hard-truncate to max_chars with an ellipsis marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Document index
# ---------------------------------------------------------------------------

class _Document:
    """A single retrievable document (wiki page or raw file)."""

    __slots__ = ("name", "path", "doc_type", "content", "relevance_score")

    def __init__(
        self,
        name:     str,
        path:     Path,
        doc_type: str,   # "wiki" | "raw"
        content:  str,
    ) -> None:
        self.name            = name
        self.path            = path
        self.doc_type        = doc_type
        self.content         = content
        self.relevance_score = 0.0

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "name":            self.name,
            "path":            str(self.path),
            "type":            self.doc_type,
            "relevance_score": round(self.relevance_score, 4),
        }


def _load_corpus(wiki_dir: Path, raw_dir: Path) -> list[_Document]:
    """Load all wiki pages and raw files into an in-memory document list."""
    docs: list[_Document] = []

    if wiki_dir.exists():
        for p in sorted(wiki_dir.iterdir()):
            if p.is_file() and p.suffix == ".md":
                try:
                    docs.append(_Document(p.stem, p, "wiki", p.read_text(encoding="utf-8")))
                except Exception as exc:
                    logger.warning("Could not read wiki page %s: %s", p, exc)

    if raw_dir.exists():
        for p in sorted(raw_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
                try:
                    docs.append(_Document(p.stem, p, "raw", p.read_text(encoding="utf-8")))
                except Exception as exc:
                    logger.warning("Could not read raw file %s: %s", p, exc)

    logger.debug("Corpus loaded: %d documents (%d wiki, %d raw).",
                 len(docs),
                 sum(1 for d in docs if d.doc_type == "wiki"),
                 sum(1 for d in docs if d.doc_type == "raw"))
    return docs


def _retrieve(
    query:          str,
    corpus:         list[_Document],
    max_sources:    int,
    runtime:        Any,
    use_embeddings: bool,
) -> list[_Document]:
    """
    Retrieve the top-N most relevant documents for a query.

    Strategy
    --------
    1. Score every document by keyword overlap with the query tokens.
    2. If use_embeddings=True and the runtime supports embed(), re-rank
       the top-2N candidates by embedding cosine similarity, then take top-N.
    3. Return sorted by final relevance_score descending.
    """
    query_tokens = _tokenize(query)

    for doc in corpus:
        doc.relevance_score = _keyword_score(query_tokens, doc.content)

    candidates = sorted(corpus, key=lambda d: d.relevance_score, reverse=True)

    if use_embeddings:
        pool = candidates[: max_sources * 2]
        try:
            query_vec = runtime.embed(query)
            for doc in pool:
                # Embed just the first 500 chars to stay within token budget
                doc_vec = runtime.embed(doc.content[:500])
                doc.relevance_score = _cosine_similarity(query_vec, doc_vec)
            candidates = sorted(pool, key=lambda d: d.relevance_score, reverse=True)
        except Exception as exc:
            logger.warning("Embedding re-rank failed (%s); falling back to keyword scores.", exc)

    top = candidates[:max_sources]
    logger.debug("Retrieved %d documents for query '%s...'.", len(top), query[:60])
    return top


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_SYSTEM_DECOMPOSE = (
    "You are a research planner inside a local multi-agent system. "
    "Given a research query, output a JSON array of 2–5 focused sub-queries "
    "that together cover the full scope of the original question. "
    "Each sub-query must be a short, specific string. "
    "Output ONLY the JSON array. No prose, no code fences, no commentary."
)

_SYSTEM_EXTRACT = (
    "You are a careful research analyst. "
    "Given a document and a sub-query, extract every factual claim in the "
    "document that is relevant to the sub-query. "
    "Output a JSON array of objects, each with keys: "
    "\"claim\" (string), \"confidence\" (\"high\", \"medium\", or \"low\"). "
    "If the document contains nothing relevant, output an empty array []. "
    "Output ONLY the JSON array. No prose, no code fences."
)

_SYSTEM_GAP = (
    "You are a research critic. "
    "Given a research query, a list of claims, and a list of source documents, "
    "identify what important aspects of the query are NOT adequately covered. "
    "Output a JSON array of short gap descriptions (strings). "
    "If everything is covered, output []. "
    "Output ONLY the JSON array. No prose, no code fences."
)

_SYSTEM_SYNTHESISE = (
    "You are a research synthesiser inside a local multi-agent system. "
    "Given a research query and a set of extracted claims with source references, "
    "write a structured research report in plain Markdown. "
    "Rules: "
    "1. Ground every assertion in the provided claims — do not hallucinate. "
    "2. Use the source name in parentheses after each assertion, e.g. (source: wiki-page-name). "
    "3. Structure: ## Summary, ## Findings (grouped by sub-query), ## Gaps. "
    "4. Be concise — prefer bullet points over prose paragraphs. "
    "Output ONLY the Markdown report."
)


def _build_extract_prompt(sub_query: str, doc: _Document, max_chars: int) -> str:
    return (
        f"Sub-query: {sub_query}\n\n"
        f"Document name: {doc.name} (type: {doc.doc_type})\n\n"
        f"Document content:\n{_truncate(doc.content, max_chars)}\n\n"
        "Extract all relevant factual claims as a JSON array."
    )


def _build_gap_prompt(query: str, claims: list[dict], sources: list[_Document]) -> str:
    claims_str  = json.dumps(claims[:40], indent=2)   # cap to avoid token blowout
    sources_str = ", ".join(d.name for d in sources) or "(none)"
    return (
        f"Research query: {query}\n\n"
        f"Sources consulted: {sources_str}\n\n"
        f"Extracted claims:\n{claims_str}\n\n"
        "Identify gaps as a JSON array of strings."
    )


def _build_synthesis_prompt(
    query:      str,
    sub_queries: list[str],
    claims:     list[dict],
    sources:    list[_Document],
) -> str:
    claims_str = json.dumps(claims, indent=2)
    sq_str     = "\n".join(f"  - {sq}" for sq in sub_queries)
    src_str    = "\n".join(
        f"  - {d.name} ({d.doc_type}, relevance={d.relevance_score:.3f})"
        for d in sources
    )
    return (
        f"Research query: {query}\n\n"
        f"Sub-queries investigated:\n{sq_str}\n\n"
        f"Sources consulted:\n{src_str}\n\n"
        f"Extracted claims:\n{claims_str}\n\n"
        "Write the structured Markdown research report."
    )


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_array(raw: str, context: str) -> list:
    """
    Extract and parse a JSON array from a model response.

    Strips code fences, finds the first [...] block, and parses it.
    Returns an empty list on any failure — callers must tolerate this.
    """
    text = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        logger.warning("No JSON array found in model output for %s.", context)
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for %s: %s", context, exc)
        return []


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    Multi-step iterative research and synthesis agent.

    Parameters
    ----------
    runtime :
        A RuntimeClient instance (FoundryRuntimeClient or compatible mock).
    project_root :
        Fallback root for resolving wiki_dir and raw_dir when not supplied
        in SubTask.context.
    """

    def __init__(
        self,
        runtime:      Any,
        project_root: Path | None = None,
    ) -> None:
        self._runtime      = runtime
        self._project_root = project_root or Path(__file__).resolve().parents[2]

    # -----------------------------------------------------------------------
    # AgentInterface — name
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "research_agent"

    # -----------------------------------------------------------------------
    # AgentInterface — can_handle
    # -----------------------------------------------------------------------

    def can_handle(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return any(kw in lowered for kw in _RESEARCH_KEYWORDS)

    # -----------------------------------------------------------------------
    # AgentInterface — run
    # -----------------------------------------------------------------------

    def run(self, subtask: SubTask) -> AgentResult:
        """
        Execute a multi-step research workflow:

          1. Validate context and resolve paths
          2. Load corpus (wiki + raw)
          3. Decompose query into sub-queries
          4. For each sub-query: retrieve → read → extract claims  (iterative)
          5. Detect gaps
          6. Synthesise final report
          7. Return AgentResult

        No stdin.  No sys.exit().  No interactive prompts.
        """
        ctx = subtask.context

        # -- 1. Validate and resolve -----------------------------------------

        query = ctx.get("query", "").strip()
        if not query:
            # Fall back to the subtask instruction if query not in context
            query = subtask.instruction.strip()
        if not query:
            return self._fail(subtask, "No query provided in context['query'] or subtask.instruction.")

        wiki_dir   = Path(ctx.get("wiki_dir",  self._project_root / "wiki"))
        raw_dir    = Path(ctx.get("raw_dir",   self._project_root / "raw"))
        max_src    = int(ctx.get("max_sources",         5))
        max_iter   = int(ctx.get("max_iterations",      3))
        max_tok    = int(ctx.get("max_tokens_per_call", 1024))
        temperature      = float(ctx.get("temperature",    0.2))
        use_embeddings   = bool(ctx.get("use_embeddings", False))

        logger.info("[%s] Starting research — query: '%s...'", self.name, query[:80])

        # -- 2. Load corpus --------------------------------------------------

        try:
            corpus = _load_corpus(wiki_dir, raw_dir)
        except Exception as exc:
            return self._fail(subtask, f"Corpus load error: {exc}")

        if not corpus:
            logger.warning("[%s] Corpus is empty — wiki_dir=%s  raw_dir=%s",
                           self.name, wiki_dir, raw_dir)

        # -- 3. Decompose query ----------------------------------------------

        sub_queries = self._decompose(query, max_tok, temperature)
        if not sub_queries:
            logger.warning("[%s] Decomposition returned no sub-queries; using original query.",
                           self.name)
            sub_queries = [query]

        logger.info("[%s] Sub-queries: %s", self.name, sub_queries)

        # -- 4. Iterative retrieve → read → extract --------------------------

        all_claims:  list[dict]      = []
        all_sources: list[_Document] = []
        seen_docs:   set[str]        = set()
        iterations                   = 0

        for iteration in range(max_iter):
            iterations = iteration + 1
            new_claims_this_round = 0

            for sq in sub_queries:
                docs = _retrieve(sq, corpus, max_src, self._runtime, use_embeddings)

                for doc in docs:
                    if doc.name in seen_docs:
                        continue
                    seen_docs.add(doc.name)

                    claims = self._extract_claims(sq, doc, max_tok, temperature)
                    if claims:
                        # Annotate each claim with its source
                        for c in claims:
                            c["source"] = doc.name
                        all_claims.extend(claims)
                        new_claims_this_round += len(claims)

                    if doc not in all_sources:
                        all_sources.append(doc)

            logger.info(
                "[%s] Iteration %d complete — new claims: %d  total: %d  docs seen: %d",
                self.name, iterations, new_claims_this_round, len(all_claims), len(seen_docs),
            )

            # Early exit — no new information found this round
            if new_claims_this_round == 0:
                logger.info("[%s] No new claims in iteration %d; stopping early.",
                            self.name, iterations)
                break

        # -- 5. Detect gaps --------------------------------------------------

        gaps = self._detect_gaps(query, all_claims, all_sources, max_tok, temperature)

        # -- 6. Synthesise ---------------------------------------------------

        report = self._synthesise(
            query, sub_queries, all_claims, all_sources, max_tok, temperature
        )

        # -- 7. Build output -------------------------------------------------

        output: dict[str, Any] = {
            "report":      report,
            "claims":      all_claims,
            "sources":     [d.to_source_dict() for d in all_sources],
            "sub_queries": sub_queries,
            "gaps":        gaps,
            "iterations":  iterations,
            "query":       query,
        }

        logger.info(
            "[%s] Complete — claims=%d  sources=%d  gaps=%d  iterations=%d",
            self.name, len(all_claims), len(all_sources), len(gaps), iterations,
        )

        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = self.name,
            status     = TaskStatus.COMPLETE,
            output     = output,
        )

    # -----------------------------------------------------------------------
    # Internal reasoning steps
    # -----------------------------------------------------------------------

    def _decompose(
        self,
        query:       str,
        max_tokens:  int,
        temperature: float,
    ) -> list[str]:
        """Ask the model to break the query into focused sub-queries."""
        try:
            raw = self._runtime.infer(
                system      = _SYSTEM_DECOMPOSE,
                prompt      = f"Research query: {query}",
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            result = _parse_json_array(raw, "decompose")
            # Validate: all items must be non-empty strings
            return [s for s in result if isinstance(s, str) and s.strip()]
        except Exception as exc:
            logger.warning("[%s] Decomposition failed: %s", self.name, exc)
            return []

    def _extract_claims(
        self,
        sub_query:   str,
        doc:         _Document,
        max_tokens:  int,
        temperature: float,
    ) -> list[dict]:
        """
        Ask the model to extract relevant claims from a single document.

        Caps document content at 4000 chars to protect the KV cache —
        the same guard Phi-4-mini needs that the WikiAgent already applies.
        """
        prompt = _build_extract_prompt(sub_query, doc, max_chars=4_000)
        try:
            raw    = self._runtime.infer(
                system      = _SYSTEM_EXTRACT,
                prompt      = prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            result = _parse_json_array(raw, f"extract/{doc.name}")
            # Validate shape: must have "claim" key; "confidence" is optional
            valid = []
            for item in result:
                if isinstance(item, dict) and isinstance(item.get("claim"), str):
                    item.setdefault("confidence", "medium")
                    valid.append(item)
            return valid
        except Exception as exc:
            logger.warning("[%s] Claim extraction failed for %s: %s",
                           self.name, doc.name, exc)
            return []

    def _detect_gaps(
        self,
        query:       str,
        claims:      list[dict],
        sources:     list[_Document],
        max_tokens:  int,
        temperature: float,
    ) -> list[str]:
        """Ask the model what the corpus failed to adequately cover."""
        if not claims and not sources:
            return ["No corpus documents found — all aspects of the query are unaddressed."]
        prompt = _build_gap_prompt(query, claims, sources)
        try:
            raw    = self._runtime.infer(
                system      = _SYSTEM_GAP,
                prompt      = prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            result = _parse_json_array(raw, "gaps")
            return [s for s in result if isinstance(s, str) and s.strip()]
        except Exception as exc:
            logger.warning("[%s] Gap detection failed: %s", self.name, exc)
            return []

    def _synthesise(
        self,
        query:       str,
        sub_queries: list[str],
        claims:      list[dict],
        sources:     list[_Document],
        max_tokens:  int,
        temperature: float,
    ) -> str:
        """Combine all extracted claims into a structured Markdown report."""
        if not claims:
            return (
                f"## Research Report: {query}\n\n"
                "No relevant information was found in the corpus for this query.\n"
            )
        prompt = _build_synthesis_prompt(query, sub_queries, claims, sources)
        try:
            return self._runtime.infer(
                system      = _SYSTEM_SYNTHESISE,
                prompt      = prompt,
                max_tokens  = max_tokens * 2,   # synthesis needs more room than extraction
                temperature = temperature,
            )
        except Exception as exc:
            logger.error("[%s] Synthesis failed: %s", self.name, exc)
            # Fallback: return a minimal report from raw claims
            lines = [f"## Research Report: {query}\n", "## Findings\n"]
            for c in claims:
                src = c.get("source", "unknown")
                lines.append(f"- {c.get('claim', '')} (source: {src})")
            return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Error helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _fail(subtask: SubTask, reason: str) -> AgentResult:
        """Construct a FAILED AgentResult without raising."""
        logger.error("[research_agent] subtask %s failed: %s", subtask.subtask_id, reason)
        return AgentResult(
            subtask_id = subtask.subtask_id,
            agent_name = "research_agent",
            status     = TaskStatus.FAILED,
            output     = {},
            error      = reason,
        )


# ---------------------------------------------------------------------------
# Protocol conformance check
# ---------------------------------------------------------------------------

def _assert_protocol_conformance() -> None:
    """Verify ResearchAgent satisfies AgentInterface at import time."""
    from controller_agent import AgentInterface

    class _MockRuntime:
        def infer(self, *a, **kw) -> str: return "[]"
        def embed(self, text: str) -> list[float]: return [0.0] * 768

    agent = ResearchAgent(runtime=_MockRuntime())
    assert isinstance(agent, AgentInterface), (
        "ResearchAgent does not satisfy the AgentInterface Protocol."
    )


# ---------------------------------------------------------------------------
# Wiring example — for reference, not executed in production
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json as _json
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, stream=sys.stdout)

    from foundry_runtime_client import FoundryRuntimeClient
    from controller_agent import ControllerAgent
    import uuid

    runtime          = FoundryRuntimeClient()
    research_agent   = ResearchAgent(runtime=runtime)

    # Run standalone (no controller) for quick iteration during development
    from controller_agent import SubTask
    result = research_agent.run(SubTask(
        subtask_id = str(uuid.uuid4()),
        agent_name = "research_agent",
        instruction = "What do we know about transformer attention mechanisms?",
        context = {
            "query":          "What do we know about transformer attention mechanisms?",
            "wiki_dir":       "/absolute/path/to/your/wiki",
            "raw_dir":        "/absolute/path/to/your/raw",
            "max_sources":    5,
            "max_iterations": 3,
            "use_embeddings": False,
        },
    ))

    print(_json.dumps(result.output, indent=2))

    # Or wire through the full Controller stack:
    #
    # controller = ControllerAgent(
    #     runtime = runtime,
    #     agents  = [research_agent, WikiAgent(runtime=runtime)],
    # )
    # output = controller.handle_task({
    #     "instruction": "Research transformer attention mechanisms.",
    #     "context": {
    #         "query":    "What do we know about transformer attention?",
    #         "wiki_dir": "/absolute/path/to/your/wiki",
    #         "raw_dir":  "/absolute/path/to/your/raw",
    #     },
    # })
    # print(_json.dumps(output, indent=2))
