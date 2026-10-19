"""
OllamaRuntimeClient — behaviour with and without a chat_model.

chat_model has no default (DEFAULT_CHAT_MODEL == "") because Ollama serves
models of wildly different size, so silently falling back to a specific one
(e.g. a multi-gigabyte local pull) is never safe. An empty chat_model still
constructs — startup and a live switch must reach health_check() so the
first-run auto-pick (§16.20) can run — but chat calls fail fast with a clear
error until a model is set.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from localist import ollama_runtime_client
from localist.ollama_runtime_client import DEFAULT_CHAT_MODEL, OllamaRuntimeClient


def test_default_chat_model_is_empty():
    assert DEFAULT_CHAT_MODEL == ""


def test_empty_chat_model_constructs():
    client = OllamaRuntimeClient()
    assert client._chat_model == ""


def test_explicit_chat_model_constructs_successfully():
    client = OllamaRuntimeClient(chat_model="gemma4:e4b-mlx")
    assert client._chat_model == "gemma4:e4b-mlx"


def test_infer_stream_without_chat_model_raises_before_any_request(monkeypatch):
    post = MagicMock(side_effect=AssertionError("no request may be sent without a chat model"))
    monkeypatch.setattr(ollama_runtime_client.requests, "post", post)

    with pytest.raises(RuntimeError, match="No Ollama chat model is configured"):
        next(OllamaRuntimeClient().infer_stream(prompt="hi"))


def test_infer_without_chat_model_raises(monkeypatch):
    monkeypatch.setattr(
        ollama_runtime_client.requests, "post",
        MagicMock(side_effect=AssertionError("no request may be sent without a chat model")),
    )

    with pytest.raises(RuntimeError, match="No Ollama chat model is configured"):
        OllamaRuntimeClient().infer(prompt="hi")


def test_health_check_without_chat_model_reports_not_found(monkeypatch):
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"models": [{"model": "gemma4:31b-cloud"}, {"model": "nomic-embed-text:latest"}]}
    monkeypatch.setattr(ollama_runtime_client.requests, "get", MagicMock(return_value=resp))

    health = OllamaRuntimeClient().health_check()

    assert health["reachable"] is True
    assert health["chat_model_found"] is False
    assert health["chat_model"] == ""
    assert health["models"] == ["gemma4:31b-cloud", "nomic-embed-text:latest"]
