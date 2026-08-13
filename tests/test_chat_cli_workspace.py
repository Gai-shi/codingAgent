from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ai_job.chat_cli import (
    APP_ROOT,
    build_initial_messages,
    default_session_record_path,
    default_trace_log_path,
    main,
    print_context,
    resolve_workspace_root,
)
from ai_job.communication import AssistantMessage, MessageState, SystemMessage, UserMessage
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
            context_window_override=None,
            compaction_reserve_tokens=16384,
            compaction_keep_recent_tokens=20000,
            system_prompt="system prompt",
            filter_terminal_log_level="none",
        )
        workspace = Path("/tmp/example-workspace")

        messages = build_initial_messages(app_env, workspace)

        self.assertEqual(len(messages), 1)
        self.assertIn("system prompt", messages[0].content)
        self.assertIn("Current workspace root: /tmp/example-workspace", messages[0].content)

    def test_context_output_uses_model_visible_history(self):
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                UserMessage(content="old"),
                UserMessage(content="active"),
                AssistantMessage(content="hidden", visible_to_model=False),
            ],
            context_start_index=2,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            print_context(message_state.model_visible_history())

        text = output.getvalue()
        self.assertIn("sys", text)
        self.assertIn("active", text)
        self.assertNotIn("old", text)
        self.assertNotIn("hidden", text)

    def test_trace_log_path_lives_under_ai_job_project_root(self):
        self.assertEqual(default_trace_log_path(), APP_ROOT / ".ai_job" / "logs" / "log.log")

    def test_session_record_path_lives_under_ai_job_project_root(self):
        self.assertEqual(
            default_session_record_path(),
            APP_ROOT / ".ai_job" / "sessions" / "sessions.md",
        )

    def test_main_starts_log_and_session_cleanup_after_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "key",
                    "OPENAI_MODEL": "model",
                    "FILTER_TERMINAL_LOG_LEVEL": "none",
                },
                clear=True,
            ):
                with patch("ai_job.chat_cli.LogWrapper.configure") as log_configure_mock:
                    with patch("ai_job.chat_cli.LogWrapper.cleanup_expired_logs_async") as log_cleanup_mock:
                        with patch("ai_job.chat_cli.SessionRecorder.configure") as session_configure_mock:
                            with patch(
                                "ai_job.chat_cli.SessionRecorder.cleanup_expired_session_records_async"
                            ) as session_cleanup_mock:
                                with patch("ai_job.chat_cli.SessionRecorder.record_session") as record_session_mock:
                                    with patch("ai_job.chat_cli.AgentRunner") as agent_runner_mock:
                                        with patch("builtins.input", side_effect=EOFError):
                                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                                exit_code = main(["--workspace", tmp_dir])

        self.assertEqual(exit_code, 0)
        log_configure_mock.assert_called_once()
        log_started_at = log_configure_mock.call_args.kwargs["session_started_at"]
        log_cleanup_mock.assert_called_once_with()
        session_configure_mock.assert_called_once()
        self.assertEqual(session_configure_mock.call_args.kwargs["session_started_at"], log_started_at)
        self.assertEqual(
            session_configure_mock.call_args.kwargs["session_path"],
            default_session_record_path(),
        )
        self.assertEqual(
            session_configure_mock.call_args.kwargs["metadata"]["workspace"],
            str(Path(tmp_dir).resolve()),
        )
        session_cleanup_mock.assert_called_once_with()
        record_session_mock.assert_called_once()
        self.assertEqual(record_session_mock.call_args.args[0], "SystemMessage")
        self.assertEqual(record_session_mock.call_args.args[2], "text")
        self.assertIn("compression_manager", agent_runner_mock.call_args.kwargs)
        self.assertIsNotNone(agent_runner_mock.call_args.kwargs["compression_manager"])


if __name__ == "__main__":
    unittest.main()
