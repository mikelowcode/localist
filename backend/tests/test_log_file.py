"""
Tests for the packaged app's log files (§7.30): localist.log_file and the
Settings "Show logs" endpoints (GET /settings/logs, POST /settings/logs/open).

Every test removes the handlers it added and resets log_file._active_dir —
install_file_logging() mutates the process-wide root logger.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from localist import log_file, main


@pytest.fixture(autouse=True)
def clean_root_logger(monkeypatch):
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    noisy_levels = {n: logging.getLogger(n).level for n in log_file._NOISY_LOGGERS}
    monkeypatch.setattr(log_file, "_active_dir", None)
    monkeypatch.delenv(log_file.LOG_DIR_ENV, raising=False)
    yield
    for h in list(root.handlers):
        if h not in before_handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(before_level)
    for n, lvl in noisy_levels.items():
        logging.getLogger(n).setLevel(lvl)


def _our_handlers():
    return [h for h in logging.getLogger().handlers if hasattr(h, "_localist_log_file")]


class TestDefaultLogDir:
    def test_default_is_library_logs_localist(self):
        assert log_file.default_log_dir() == log_file.Path.home() / "Library" / "Logs" / "Localist"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(log_file.LOG_DIR_ENV, str(tmp_path / "x"))
        assert log_file.default_log_dir() == tmp_path / "x"


class TestInstallFileLogging:
    def test_creates_folder_and_writes_lines(self, tmp_path):
        d = tmp_path / "Logs" / "Localist"
        path = log_file.install_file_logging("backend", d)

        assert path == d / "backend.log"
        logging.getLogger("localist.test").info("hello from the test")
        for h in _our_handlers():
            h.flush()
        text = path.read_text()
        assert "hello from the test" in text
        # Full date in the file (the terminal format only has the time).
        assert text[:4].isdigit() and text[4] == "-"
        assert log_file.active_log_dir() == d

    def test_rotation_is_size_capped(self, tmp_path):
        log_file.install_file_logging("mcp", tmp_path)
        (h,) = _our_handlers()
        assert isinstance(h, RotatingFileHandler)
        assert h.maxBytes == 5 * 1024 * 1024
        assert h.backupCount == 3

    def test_idempotent_per_name(self, tmp_path):
        log_file.install_file_logging("backend", tmp_path)
        log_file.install_file_logging("backend", tmp_path)
        assert len(_our_handlers()) == 1

    def test_uses_env_dir_when_none_given(self, monkeypatch, tmp_path):
        monkeypatch.setenv(log_file.LOG_DIR_ENV, str(tmp_path))
        assert log_file.install_file_logging("backend") == tmp_path / "backend.log"

    def test_unwritable_folder_returns_none_without_raising(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("not a folder")
        assert log_file.install_file_logging("backend", blocker / "sub") is None
        assert _our_handlers() == []
        assert log_file.active_log_dir() is None

    def test_root_level_raised_to_info_from_default_warning(self, tmp_path):
        logging.getLogger().setLevel(logging.WARNING)
        log_file.install_file_logging("backend", tmp_path)
        assert logging.getLogger().level == logging.INFO

    def test_existing_debug_level_kept(self, tmp_path):
        logging.getLogger().setLevel(logging.DEBUG)
        log_file.install_file_logging("backend", tmp_path)
        assert logging.getLogger().level == logging.DEBUG

    def test_per_event_trace_loggers_quieted(self, tmp_path):
        logging.getLogger("sse_starlette").setLevel(logging.DEBUG)
        log_file.install_file_logging("mcp", tmp_path)
        assert logging.getLogger("sse_starlette").level == logging.INFO


class TestApplyConfiguredLevel:
    def test_no_op_without_file_logging(self):
        logging.getLogger().setLevel(logging.WARNING)
        log_file.apply_configured_level("DEBUG")
        assert logging.getLogger().level == logging.WARNING

    def test_applies_once_file_logging_installed(self, tmp_path):
        log_file.install_file_logging("backend", tmp_path)
        log_file.apply_configured_level("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        assert logging.getLogger("sse_starlette").level == logging.INFO


class TestListLogFiles:
    def test_lists_current_and_rotated_newest_first(self, tmp_path):
        for i, name in enumerate(["backend.log", "backend.log.1", "app_2026-11-11_12-00-00.log", "notes.txt"]):
            p = tmp_path / name
            p.write_text("x" * (i + 1))
            os.utime(p, (1_700_000_000 + i, 1_700_000_000 + i))

        files = log_file.list_log_files(tmp_path)

        assert [f["name"] for f in files] == [
            "app_2026-11-11_12-00-00.log", "backend.log.1", "backend.log",
        ]
        assert files[-1]["size"] == 1

    def test_missing_folder_is_empty(self, tmp_path):
        assert log_file.list_log_files(tmp_path / "nope") == []


class TestLogsEndpoints:
    @pytest.fixture()
    def client(self, monkeypatch, tmp_path):
        monkeypatch.setenv(log_file.LOG_DIR_ENV, str(tmp_path))
        return TestClient(main.app), tmp_path

    def test_status_from_source_reports_not_enabled(self, client):
        test_client, d = client
        (d / "backend.log").write_text("line\n")

        data = test_client.get("/settings/logs").json()

        assert data["enabled"] is False
        assert [f["name"] for f in data["files"]] == ["backend.log"]

    def test_status_enabled_when_file_logging_installed(self, client):
        test_client, d = client
        log_file.install_file_logging("backend", d)

        data = test_client.get("/settings/logs").json()

        assert data["enabled"] is True
        assert "backend.log" in [f["name"] for f in data["files"]]

    def test_home_shown_as_tilde(self, monkeypatch):
        monkeypatch.delenv(log_file.LOG_DIR_ENV, raising=False)
        data = TestClient(main.app).get("/settings/logs").json()
        assert data["dir"] == "~/Library/Logs/Localist"

    def test_open_runs_finder_open_on_the_folder(self, client, monkeypatch):
        test_client, d = client
        monkeypatch.setattr(main.sys, "platform", "darwin")
        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        with patch.object(main.asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)) as spawn:
            resp = test_client.post("/settings/logs/open")

        assert resp.status_code == 200
        assert spawn.call_args.args == ("open", str(d))

    def test_open_missing_folder_is_404(self, monkeypatch, tmp_path):
        monkeypatch.setenv(log_file.LOG_DIR_ENV, str(tmp_path / "missing"))
        monkeypatch.setattr(main.sys, "platform", "darwin")
        resp = TestClient(main.app).post("/settings/logs/open")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "No log files yet."

    def test_open_not_supported_off_macos(self, client, monkeypatch):
        test_client, _ = client
        monkeypatch.setattr(main.sys, "platform", "linux")
        assert test_client.post("/settings/logs/open").status_code == 501
