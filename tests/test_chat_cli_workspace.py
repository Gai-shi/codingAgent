from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_job.chat_cli import (
    APP_ROOT,
    default_session_record_path,
    default_trace_log_path,
    main,
    print_context,
    resolve_workspace_root,
)
from ai_job.communication import AssistantMessage, MessageState, SystemMessage, UserMessage
from ai_job.tools import ToolRegistry


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

    def test_main_starts_session_lifecycle_and_records_exit_snapshot(self):
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
                runtime = SimpleNamespace(
                    message_state=MessageState(history=[SystemMessage(content="system prompt")]),
                    tool_registry=ToolRegistry([]),
                    agent_runner=SimpleNamespace(run_turn=lambda _message_state: "unused"),
                )
                lifecycle = Mock()
                with patch("ai_job.chat_cli.CliSessionLifecycle", return_value=lifecycle) as lifecycle_class_mock:
                    with patch(
                        "ai_job.chat_cli.create_cli_runtime",
                        return_value=runtime,
                    ) as create_runtime_mock:
                        with patch("builtins.input", side_effect=EOFError):
                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                exit_code = main(["--workspace", tmp_dir])

        self.assertEqual(exit_code, 0)
        lifecycle_class_mock.assert_called_once_with(
            trace_log_path=default_trace_log_path(),
            session_record_path=default_session_record_path(),
        )
        lifecycle.start.assert_called_once()
        self.assertEqual(lifecycle.start.call_args.kwargs["workspace_root"], Path(tmp_dir).resolve())
        lifecycle.record_initial_system_message.assert_called_once_with(runtime.message_state)
        lifecycle.record_exit_snapshot.assert_called_once_with(runtime.message_state)
        self.assertEqual(
            lifecycle.start.call_args.kwargs["app_env"],
            create_runtime_mock.call_args.kwargs["app_env"],
        )
        create_runtime_mock.assert_called_once()
        self.assertEqual(
            create_runtime_mock.call_args.kwargs["workspace_root"],
            Path(tmp_dir).resolve(),
        )
        self.assertTrue(callable(create_runtime_mock.call_args.kwargs["request_protected_grep_approval"]))
        self.assertTrue(create_runtime_mock.call_args.kwargs["include_compress_tool"])

    def test_main_can_disable_compress_tool_registration(self):
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
                runtime = SimpleNamespace(
                    message_state=MessageState(history=[SystemMessage(content="system prompt")]),
                    tool_registry=ToolRegistry([]),
                    agent_runner=SimpleNamespace(run_turn=lambda _message_state: "unused"),
                )
                with patch("ai_job.chat_cli.CliSessionLifecycle", return_value=Mock()):
                    with patch(
                        "ai_job.chat_cli.create_cli_runtime",
                        return_value=runtime,
                    ) as create_runtime_mock:
                        with patch("builtins.input", side_effect=EOFError):
                            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                                exit_code = main(
                                    [
                                        "--workspace",
                                        tmp_dir,
                                        "--disable-compress-tool",
                                    ]
                                )

        self.assertEqual(exit_code, 0)
        self.assertFalse(create_runtime_mock.call_args.kwargs["include_compress_tool"])


if __name__ == "__main__":
    unittest.main()
