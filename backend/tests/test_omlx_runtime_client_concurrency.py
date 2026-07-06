"""
oMLX runtime client — overlap-detection logging + serialization lock.

Covers the two concerns added to omlx_runtime_client.py:
  1. RUNTIME_OVERLAP warning logging, correlated by `label`, whenever a
     call starts while another is already in flight.
  2. The module-level threading.Lock that brackets the HTTP call in
     infer_stream() (and, transitively, infer()) so two overlapping
     call sites (main_dispatch / implicit_extraction / working_state)
     can never both be talking to the single oMLX server at once.

Call sites into OMLXRuntimeClient are plain synchronous methods (the
FastAPI layer runs them in a worker thread via asyncio.to_thread), so the
lock under test is threading.Lock, not asyncio.Lock — see the module
docstring in omlx_runtime_client.py for why.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from unittest.mock import patch

import omlx_runtime_client as _omlx_mod
from omlx_runtime_client import OMLXRuntimeClient


def _fake_response(chunks: list[str], delay_s: float = 0.0):
    """Build a fake requests.Response-like object for _iter_sse_chunks."""

    class _FakeResponse:
        status_code = 200

        def iter_lines(self, decode_unicode: bool = True):
            for chunk in chunks:
                if delay_s:
                    time.sleep(delay_s)
                envelope = {"choices": [{"delta": {"content": chunk}}]}
                yield f"data: {json.dumps(envelope)}"
            yield "data: [DONE]"

    return _FakeResponse()


class TestOverlapWarningLogging:
    def setup_method(self):
        # Guard against cross-test leakage if a prior test left the
        # counter non-zero due to an assertion failure mid-call.
        _omlx_mod._inflight_count = 0

    def test_no_warning_when_no_call_in_flight(self, caplog):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
            with caplog.at_level(logging.WARNING, logger="omlx_runtime_client"):
                result = client.infer(prompt="hello", label="main_dispatch")

        assert result == "hi"
        assert "RUNTIME_OVERLAP" not in caplog.text
        assert _omlx_mod._inflight_count == 0

    def test_warning_fires_and_names_label_when_counter_already_positive(self, caplog):
        client = OMLXRuntimeClient()
        # Simulate a call already in flight (e.g. a prior turn's background
        # write that hasn't finished) without needing real concurrency.
        _omlx_mod._inflight_count = 1
        try:
            with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
                with caplog.at_level(logging.WARNING, logger="omlx_runtime_client"):
                    client.infer(prompt="hello", label="implicit_extraction")
        finally:
            pass

        assert "RUNTIME_OVERLAP detected" in caplog.text
        assert "label=implicit_extraction" in caplog.text
        # Decremented back to the pre-existing in-flight call's count (1),
        # not to 0 — this call didn't own that other in-flight slot.
        assert _omlx_mod._inflight_count == 1
        _omlx_mod._inflight_count = 0

    def test_counter_decrements_on_exception(self):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", side_effect=_omlx_mod.requests.ConnectionError("down")):
            try:
                client.infer(prompt="hello", label="working_state")
            except RuntimeError:
                pass

        assert _omlx_mod._inflight_count == 0


class _NullLock:
    """No-op stand-in for threading.Lock — lets two threads run the
    'locked' section concurrently, used only to prove the test below is
    actually capable of detecting overlap (rather than passing trivially)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestSerializationLock:
    def setup_method(self):
        _omlx_mod._inflight_count = 0

    def _concurrent_calls(self, client, delay_s: float):
        def _slow_post(*args, **kwargs):
            return _fake_response(["tok"], delay_s=delay_s)

        def _call(label: str):
            client.infer(prompt="hello", label=label)

        with patch.object(_omlx_mod.requests, "post", side_effect=_slow_post):
            t1 = threading.Thread(target=_call, args=("main_dispatch",))
            t2 = threading.Thread(target=_call, args=("implicit_extraction",))
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

    def test_lock_serializes_concurrent_infer_calls(self, caplog):
        """
        Two threads (standing in for two call sites racing against the same
        oMLX server) must never trigger a RUNTIME_OVERLAP warning once the
        lock is in place — the second thread blocks on _inflight_lock until
        the first has fully finished (and decremented), so its own overlap
        check always sees 0.
        """
        client = OMLXRuntimeClient()
        with caplog.at_level(logging.WARNING, logger="omlx_runtime_client"):
            self._concurrent_calls(client, delay_s=0.05)

        assert "RUNTIME_OVERLAP" not in caplog.text
        assert _omlx_mod._inflight_count == 0

    def test_sanity_overlap_is_detected_when_lock_disabled(self, caplog):
        """
        Proves the harness above isn't just insensitive to overlap: with
        the real lock swapped for a no-op, the same two concurrent calls
        DO trigger RUNTIME_OVERLAP (and the raw race can corrupt the
        counter, which is exactly why the lock — not just the counter's
        own bookkeeping — must bracket the call).
        """
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod, "_inflight_lock", _NullLock()):
            with caplog.at_level(logging.WARNING, logger="omlx_runtime_client"):
                self._concurrent_calls(client, delay_s=0.05)

        assert "RUNTIME_OVERLAP" in caplog.text
        _omlx_mod._inflight_count = 0

    def test_label_reaches_debug_log(self, caplog):
        client = OMLXRuntimeClient()
        with patch.object(_omlx_mod.requests, "post", return_value=_fake_response(["hi"])):
            with caplog.at_level(logging.DEBUG, logger="omlx_runtime_client"):
                client.infer(prompt="hello", label="working_state")

        assert "label=working_state" in caplog.text
