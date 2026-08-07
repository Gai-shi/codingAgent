"""Provider-agnostic agent turn runner."""

from __future__ import annotations

from dataclasses import asdict

from ..communication import MessageHistory, ToolMessage
from ..infra.logging import LogWrapper
from ..infra.session_recording import SessionRecorder
from ..provider_adapters import BaseChatModel
from ..tools import ToolCall, ToolExecutor, ToolRegistry


TRACE_TAG = "trace"


class AgentRunner:
    """Run one user turn, including zero or more native tool-calling rounds."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        max_tool_rounds: int,
    ) -> None:
        self._chat_model = chat_model
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._max_tool_rounds = max_tool_rounds

    def run_turn(self, history: MessageHistory) -> str:
        """Run the agent loop for one user turn and mutate history in-place."""
        for round_index in range(self._max_tool_rounds):
            round_number = round_index + 1
            LogWrapper.debug(TRACE_TAG, f"round={round_number}")

            assistant_message = self._chat_model.complete(history, self._tool_registry)
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
                tool_content = self._tool_executor.execute(tool_call)
                history.append(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        content=tool_content,
                    )
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
