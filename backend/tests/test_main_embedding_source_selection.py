"""
main._configure_embedding_source() — two-tier embedding source precedence
(runtime-backend embed, or keyword-only).

This is a mocked/forced-condition test, not real cross-platform execution.
lifespan() itself is never exercised (no existing pattern in this suite
triggers the real FastAPI lifespan — see test_main_memory_episodes.py's
docstring), so the branch selection was pulled out into
_configure_embedding_source() specifically so it's callable and assertable
in isolation, without needing to run the full startup sequence (real runtime
construction, directory indexing, graph build, etc.).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from localist import main


def _settings(*, embedding_model="", runtime_backend="ollama"):
    return SimpleNamespace(
        embedding_model=embedding_model,
        runtime_backend=runtime_backend,
    )


def _runtime():
    return SimpleNamespace(embed=MagicMock(name="runtime.embed"))


class TestRuntimeBackendTier:
    def test_runtime_embed_selected_when_configured_and_found(self, caplog):
        settings = _settings(embedding_model="nomic-embed-text:latest")
        runtime = _runtime()
        health = {"embed_model_found": True}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn = main._configure_embedding_source(settings, runtime, health)

        assert embed_fn is runtime.embed
        assert "Runtime-backend embeddings ready" in caplog.text

    def test_not_found_falls_back_to_keyword_only(self, caplog):
        settings = _settings(embedding_model="nomic-embed-text:latest")
        runtime = _runtime()
        health = {"embed_model_found": False}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn = main._configure_embedding_source(settings, runtime, health)

        assert embed_fn is None
        assert "keyword-only" in caplog.text


class TestKeywordOnlyTier:
    def test_no_model_configured_returns_none(self, caplog):
        settings = _settings(embedding_model="")
        runtime = _runtime()
        health = {"embed_model_found": False}

        with caplog.at_level(logging.INFO, logger="localist.main"):
            embed_fn = main._configure_embedding_source(settings, runtime, health)

        assert embed_fn is None
        assert "keyword-only" in caplog.text


class TestDeriveActiveEmbeddingModelName:
    """
    main._derive_active_embedding_model_name() — the two-tier name
    derivation that feeds Planner's per-gate threshold resolution
    (resolve_gate_tiers(), docs/architecture/16-runtime-backend-layer.md
    §16.4). Mirrors _configure_embedding_source()'s own tier precedence.
    """

    def test_tier1_runtime_backend_embed_returns_settings_model(self):
        settings = _settings(embedding_model="nomic-embed-text:latest")
        embed_fn = MagicMock(name="runtime.embed")

        name = main._derive_active_embedding_model_name(settings, embed_fn)

        assert name == "nomic-embed-text:latest"

    def test_tier2_keyword_only_returns_none(self):
        settings = _settings(embedding_model="")

        name = main._derive_active_embedding_model_name(settings, None)

        assert name is None
