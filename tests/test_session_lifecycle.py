from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ai_job.communication import MessageState, SystemMessage, ToolMessage
from ai_job.composition import CliSessionLifecycle
from ai_job.infra.env import AppEnv


def make_app_env() -> AppEnv:
    return AppEnv(
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


class CliSessionLifecycleTest(unittest.TestCase):
    def test_start_configures_log_and_session_recorders(self):
        lifecycle = CliSessionLifecycle(
            trace_log_path=Path("/tmp/ai-job/log.log"),
            session_record_path=Path("/tmp/ai-job/sessions.md"),
        )
        started_at = datetime(2026, 8, 13, 17, 44, 43)

        with patch("ai_job.composition.session_lifecycle.LogWrapper.configure") as log_configure_mock:
            with patch(
                "ai_job.composition.session_lifecycle.LogWrapper.cleanup_expired_logs_async"
            ) as log_cleanup_mock:
                with patch(
                    "ai_job.composition.session_lifecycle.SessionRecorder.configure"
                ) as session_configure_mock:
                    with patch(
                        "ai_job.composition.session_lifecycle.SessionRecorder.cleanup_expired_session_records_async"
                    ) as session_cleanup_mock:
                        lifecycle.start(
                            app_env=make_app_env(),
                            workspace_root=Path("/tmp/workspace"),
                            session_started_at=started_at,
                        )

        log_configure_mock.assert_called_once_with(
            log_path=Path("/tmp/ai-job/log.log"),
            filter_terminal_log_level="none",
            session_started_at=started_at,
        )
        session_configure_mock.assert_called_once_with(
            session_path=Path("/tmp/ai-job/sessions.md"),
            session_started_at=started_at,
            metadata={
                "workspace": "/tmp/workspace",
                "model": "model",
                "base_url": "http://localhost:8787/v1",
            },
        )
        log_cleanup_mock.assert_called_once_with()
        session_cleanup_mock.assert_called_once_with()

    def test_record_initial_system_message_writes_system_section(self):
        lifecycle = CliSessionLifecycle(
            trace_log_path=Path("/tmp/log.log"),
            session_record_path=Path("/tmp/sessions.md"),
        )
        message_state = MessageState(history=[SystemMessage(content="system prompt")])

        with patch("ai_job.composition.session_lifecycle.SessionRecorder.record_session") as record_mock:
            lifecycle.record_initial_system_message(message_state)

        record_mock.assert_called_once_with("SystemMessage", "system prompt", "text")

    def test_record_exit_snapshot_writes_raw_and_model_visible_history(self):
        lifecycle = CliSessionLifecycle(
            trace_log_path=Path("/tmp/log.log"),
            session_record_path=Path("/tmp/sessions.md"),
        )
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

        with patch("ai_job.composition.session_lifecycle.SessionRecorder.record_session") as record_mock:
            lifecycle.record_exit_snapshot(message_state)

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


if __name__ == "__main__":
    unittest.main()
