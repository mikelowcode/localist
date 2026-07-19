"""
`is_local` (base_runtime_client.py) — added 2026-07-18 alongside
context_profile.py to let every backend-tier-aware ceiling key off one
flag instead of three ad-hoc special cases.

- oMLX only ever runs on-device: always True.
- Foundry is local-only in this deployment (its own module docstring says
  "local only, never cloud"): always True.
- Ollama can serve either a local pull or an Ollama-Cloud-hosted model
  through the *same* local daemon (base_url stays localhost:11434 either
  way — see docs/architecture/16-runtime-backend-layer.md §16.4's
  live-verified "gemma4:31b-cloud, proxied through ollama.com" config), so
  its is_local derivation is covered separately and more thoroughly in
  tests/test_ollama_runtime_client.py::TestIsLocalAndNumCtx — this file
  only re-confirms the two straightforward cases for completeness.

Also covers context_profile.py's ContextProfile dataclass values directly,
since every other test in this session exercises them only indirectly
through ControllerAgent/OllamaRuntimeClient.
"""

from __future__ import annotations

from omlx_runtime_client import OMLXRuntimeClient
from foundry_runtime_client import FoundryRuntimeClient
from ollama_runtime_client import OllamaRuntimeClient
from context_profile import LOCAL_PROFILE, CLOUD_PROFILE, ContextProfile, profile_for


class TestIsLocalPerBackend:

    def test_omlx_is_always_local(self):
        assert OMLXRuntimeClient().is_local is True

    def test_foundry_is_always_local(self):
        client = FoundryRuntimeClient(base_url="http://127.0.0.1:59999")
        assert client.is_local is True

    def test_ollama_local_model(self):
        client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
        assert client.is_local is True

    def test_ollama_cloud_model(self):
        client = OllamaRuntimeClient(chat_model="gemma4:31b-cloud")
        assert client.is_local is False


class TestContextProfileValues:

    def test_local_profile_matches_pre_existing_hardcoded_behavior(self):
        """
        These three numbers were previously hardcoded directly at the
        controller_agent.py call site (limit=5, max_tokens=300) and in
        prompt_builder.py (_CEIL_WORKING=300). LOCAL_PROFILE must not
        drift from them — that would be a silent behavior change for
        every local (oMLX/Foundry/local-Ollama) user.
        """
        assert LOCAL_PROFILE.working_memory_tokens == 300
        assert LOCAL_PROFILE.working_memory_limit  == 5

    def test_cloud_profile_removes_turn_cap(self):
        assert CLOUD_PROFILE.working_memory_limit is None

    def test_cloud_profile_budget_larger_than_local(self):
        assert CLOUD_PROFILE.working_memory_tokens > LOCAL_PROFILE.working_memory_tokens
        assert CLOUD_PROFILE.total_context_tokens  > LOCAL_PROFILE.total_context_tokens

    def test_cloud_total_context_leaves_headroom_under_128k(self):
        """
        128K is Gemma-4-31B's stated ceiling, not a safe operating budget —
        total_context_tokens must reserve room for output generation and
        the other prompt slots, not spend the literal number.
        """
        assert CLOUD_PROFILE.total_context_tokens < 128_000

    def test_cloud_working_memory_fits_under_total_budget(self):
        """working_memory_tokens must leave room for the other slots
        (session files, RAG, tool results, etc.) within the same request —
        it can't consume the entire total_context_tokens budget alone."""
        assert CLOUD_PROFILE.working_memory_tokens < CLOUD_PROFILE.total_context_tokens

    def test_profile_for_local(self):
        assert profile_for(True) is LOCAL_PROFILE

    def test_profile_for_cloud(self):
        assert profile_for(False) is CLOUD_PROFILE

    def test_context_profile_is_frozen(self):
        profile = ContextProfile(working_memory_tokens=1, working_memory_limit=1, total_context_tokens=1)
        try:
            profile.working_memory_tokens = 2  # type: ignore[misc]
            assert False, "ContextProfile must be immutable"
        except AttributeError:
            pass
