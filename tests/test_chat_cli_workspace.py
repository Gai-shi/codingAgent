from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_job.chat_cli import build_initial_messages, resolve_workspace_root, trace_log_path_for_workspace
from ai_job.infra.env import AppEnv


class ChatCliWorkspaceTest(unittest.TestCase):
    def test_resolve_workspace_root_uses_explicit_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = resolve_workspace_root(tmp_dir)

        self.assertEqual(workspace, Path(tmp_dir).resolve())

    def test_resolve_workspace_root_defaults_to_process_cwd(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.chdir(tmp_dir)
                workspace = resolve_workspace_root(None)
            finally:
                os.chdir(original_cwd)

        self.assertEqual(workspace, Path(tmp_dir).resolve())

    def test_resolve_workspace_root_rejects_missing_or_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            file_path = workspace / "file.txt"
            file_path.write_text("text", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "workspace 不存在"):
                resolve_workspace_root(str(workspace / "missing"))
            with self.assertRaisesRegex(ValueError, "workspace 不是目录"):
                resolve_workspace_root(str(file_path))

    def test_build_initial_messages_includes_workspace_root(self):
        app_env = AppEnv(
            openai_api_key="key",
            openai_model="model",
            openai_base_url="http://localhost:8787/v1",
            timeout_seconds=60.0,
            max_tool_rounds=8,
            system_prompt="system prompt",
            filter_terminal_log_level="none",
        )
        workspace = Path("/tmp/example-workspace")

        messages = build_initial_messages(app_env, workspace)

        self.assertEqual(len(messages), 1)
        self.assertIn("system prompt", messages[0].content)
        self.assertIn("Current workspace root: /tmp/example-workspace", messages[0].content)

    def test_trace_log_path_lives_under_workspace(self):
        workspace = Path("/tmp/example-workspace")

        self.assertEqual(trace_log_path_for_workspace(workspace), workspace / ".ai_job" / "trace.log")


if __name__ == "__main__":
    unittest.main()
