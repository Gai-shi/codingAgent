"""Provider-agnostic agent turn runner."""

from __future__ import annotations

from dataclasses import asdict

from ..compress import CompressionManager
from ..communication import MessageState, ToolMessage
from ..infra.logging import LogWrapper
from ..infra.session_recording import SessionRecorder
from ..provider_adapters import BaseChatModel
from ..tools import ToolCall, ToolExecutionContext, ToolExecutor, ToolRegistry
from .message_visibility import MessageVisibilityManager


TRACE_TAG = "trace"


class AgentRunner:
    """Run one user turn, including zero or more native tool-calling rounds."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        max_tool_rounds: int,
        compression_manager: CompressionManager | None = None,
        message_visibility_manager: MessageVisibilityManager | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._max_tool_rounds = max_tool_rounds
        self._compression_manager = compression_manager
        self._message_visibility_manager = message_visibility_manager or MessageVisibilityManager()

    def run_turn(self, message_state: MessageState) -> str:
        """Run the agent loop for one user turn and mutate history in-place."""
        history = message_state.history
        for round_index in range(self._max_tool_rounds):
            round_number = round_index + 1
            LogWrapper.debug(TRACE_TAG, f"round={round_number}")

            if self._compression_manager is not None:
                self._compression_manager.compress_if_needed(message_state)
            assistant_message = self._chat_model.complete(
                message_state.model_visible_history(),
                self._tool_registry,
            )
            history.append(assistant_message)
            SessionRecorder.record_session(
                "AssistantMessage",
                {
                    "content": assistant_message.content,
                    "tool_calls": [asdict(tool_call) for tool_call in assistant_message.tool_calls],
                },
                "json",
            )

            if not assistant_message.tool_calls:
                if not isinstance(assistant_message.content, str):
                    raise RuntimeError("LLM 最终响应缺少文本 content")
                return assistant_message.content

            assistant_message_index = len(history) - 1
            tool_call_count = len(assistant_message.tool_calls)
            for tool_call_index, tool_call in enumerate(assistant_message.tool_calls, start=1):
                LogWrapper.debug(
                    TRACE_TAG,
                    self._tool_call_log_line(
                        round_number=round_number,
                        tool_call_index=tool_call_index,
                        tool_call_count=tool_call_count,
                        tool_call=tool_call,
                    ),
                )
                SessionRecorder.record_session(f"ToolCall {tool_call.name}", asdict(tool_call), "json")
                tool_content = self._tool_executor.execute(
                    tool_call,
                    ToolExecutionContext(message_state=message_state),
                )
                tool_message_index = len(history)
                history.append(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        content=tool_content,
                    )
                )
                self._message_visibility_manager.apply_after_tool_execution(
                    history,
                    assistant_message_index=assistant_message_index,
                    tool_message_index=tool_message_index,
                    tool_name=tool_call.name,
                    success=tool_content == "Success",
                )
                SessionRecorder.record_session(f"ToolResult {tool_call.name}", tool_content, "text")

        raise RuntimeError(f"工具调用轮数超过上限：{self._max_tool_rounds}")
    @staticmethod
    def _tool_call_log_line(
        round_number: int,
        tool_call_index: int,
        tool_call_count: int,
        tool_call: ToolCall,
    ) -> str:
        parts = [
            f"round={round_number}",
            f"tool_call={tool_call_index}/{tool_call_count}",
            f"tool={tool_call.name}",
        ]
        path = tool_call.arguments.get("path")
        if isinstance(path, str):
            parts.append(f"path={path}")
        return " ".join(parts)
