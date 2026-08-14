"""CLI session lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..communication import MessageState, message_history_to_debug_dicts
from ..infra.env import AppEnv
from ..infra.logging import LogWrapper
from ..infra.session_recording import SessionRecorder


@dataclass(frozen=True)
class CliSessionLifecycle:
    trace_log_path: Path
    session_record_path: Path

    def start(
        self,
        *,
        app_env: AppEnv,
        workspace_root: Path,
        session_started_at: datetime | None = None,
    ) -> None:
        started_at = session_started_at or datetime.now()
        LogWrapper.configure(
            log_path=self.trace_log_path,
            filter_terminal_log_level=app_env.filter_terminal_log_level,
            session_started_at=started_at,
        )
        SessionRecorder.configure(
            session_path=self.session_record_path,
            session_started_at=started_at,
            metadata={
                "workspace": str(workspace_root),
                "model": app_env.openai_model,
                "base_url": app_env.openai_base_url,
            },
        )
        LogWrapper.cleanup_expired_logs_async()
        SessionRecorder.cleanup_expired_session_records_async()

    def record_initial_system_message(self, message_state: MessageState) -> None:
        SessionRecorder.record_session("SystemMessage", message_state.history[0].content, "text")

    def record_exit_snapshot(self, message_state: MessageState) -> None:
        SessionRecorder.record_session(
            "MessageState raw",
            {
                "context_start_index": message_state.context_start_index,
                "history": message_history_to_debug_dicts(message_state.history),
            },
            "json",
        )
        SessionRecorder.record_session(
            "MessageState model_visible",
            {
                "context_start_index": message_state.context_start_index,
                "history": message_history_to_debug_dicts(
                    message_state.model_visible_history(),
                    use_model_visible_content=True,
                ),
            },
            "json",
        )
