from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_job.chat_cli import (
    APP_ROOT,
    default_session_record_path,
    default_trace_log_path,
    main,
    print_context,
    record_message_state_snapshot,
    resolve_workspace_root,
)
from ai_job.communication import AssistantMessage, MessageState, SystemMessage, ToolMessage, UserMessage
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

    def test_record_message_state_snapshot_writes_raw_and_model_visible_history(self):
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                ToolMessage(
                    tool_call_id="call-1",
                    content="raw tool output",
                    compressions=["compressed tool output"],
                ),
            ]
        )

        with patch("ai_job.chat_cli.SessionRecorder.record_session") as record_mock:
            record_message_state_snapshot(message_state)

        self.assertEqual(record_mock.call_count, 2)
        raw_call = record_mock.call_args_list[0]
        visible_call = record_mock.call_args_list[1]
        self.assertEqual(raw_call.args[0], "MessageState raw")
        self.assertEqual(visible_call.args[0], "MessageState model_visible")
        self.assertEqual(raw_call.args[1]["history"][1]["content"], "raw tool output")
        self.assertEqual(
            visible_call.args[1]["history"][1]["content"],
            "compressed tool output",
        )

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
                                    runtime = SimpleNamespace(
                                        message_state=MessageState(
                                            history=[SystemMessage(content="system prompt")]
                                        ),
                                        tool_registry=ToolRegistry([]),
                                        agent_runner=SimpleNamespace(run_turn=lambda _message_state: "unused"),
                                    )
                                    with patch(
                                        "ai_job.chat_cli.create_cli_runtime",
                                        return_value=runtime,
                                    ) as create_runtime_mock:
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
        self.assertEqual(record_session_mock.call_count, 3)
        self.assertEqual(record_session_mock.call_args_list[0].args[0], "SystemMessage")
        self.assertEqual(record_session_mock.call_args_list[0].args[2], "text")
        self.assertEqual(record_session_mock.call_args_list[1].args[0], "MessageState raw")
        self.assertEqual(record_session_mock.call_args_list[2].args[0], "MessageState model_visible")
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
                with patch("ai_job.chat_cli.LogWrapper.configure"):
                    with patch("ai_job.chat_cli.LogWrapper.cleanup_expired_logs_async"):
                        with patch("ai_job.chat_cli.SessionRecorder.configure"):
                            with patch("ai_job.chat_cli.SessionRecorder.cleanup_expired_session_records_async"):
                                with patch("ai_job.chat_cli.SessionRecorder.record_session"):
                                    runtime = SimpleNamespace(
                                        message_state=MessageState(
                                            history=[SystemMessage(content="system prompt")]
                                        ),
                                        tool_registry=ToolRegistry([]),
                                        agent_runner=SimpleNamespace(run_turn=lambda _message_state: "unused"),
                                    )
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
