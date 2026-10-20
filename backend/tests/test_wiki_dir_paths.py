"""
The configured wiki folder (LOCALIST_WIKI_DIR) decides where MEMORY.md is
written and where the user profile is read — both used to be fixed at
<backend root>/wiki regardless of the setting (2026-10-20), so a backend
pointed at another wiki folder kept a separate MEMORY.md and profile.
"""

from __future__ import annotations

from pathlib import Path

from localist import main
from localist.controller_agent import _MEMORY_MD_PATH, ControllerAgent
from localist.memory_manager import MemoryManager
from localist.paths import get_backend_root
from tests.test_controller_phase4 import make_conv_agent, make_runtime


def _controller(tmp_path, **kwargs) -> ControllerAgent:
    return ControllerAgent(runtime=make_runtime(), agents=[make_conv_agent()],
                           memory_manager=MemoryManager(db_path=tmp_path / "w.db"), **kwargs)


class TestControllerFollowsWikiDir:
    def test_memory_md_and_profile_live_in_the_configured_wiki_dir(self, tmp_path):
        wiki = tmp_path / "elsewhere" / "wiki"
        (wiki / "users").mkdir(parents=True)
        (wiki / "users" / "michael.md").write_text("## Identity\n- Lives on the moon\n")
        ctrl = _controller(tmp_path, wiki_dir=wiki)
        ctrl._embed = lambda text: [1.0, 0.0]

        assert ctrl._memory_md_path == wiki / "MEMORY.md"
        ctrl._load_user_profile()
        assert ctrl._profile_lines == ["Lives on the moon"]

    def test_default_is_the_backend_root_wiki(self, tmp_path):
        ctrl = _controller(tmp_path)
        assert ctrl._wiki_dir == get_backend_root() / "wiki"
        assert ctrl._memory_md_path == _MEMORY_MD_PATH


class TestMainMemoryMdPath:
    def test_follows_state_wiki_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main._state, "wiki_dir", tmp_path / "wiki")
        assert main._memory_md_path() == tmp_path / "wiki" / "MEMORY.md"

    def test_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setattr(main._state, "wiki_dir", None)
        assert main._memory_md_path() == Path(_MEMORY_MD_PATH)
