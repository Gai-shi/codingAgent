"""LangGraph-based agent runner."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage as LangChainAIMessage
from langgraph.graph import END, StateGraph

from ...communication import AssistantMessage, MessageState, ToolMessage
from ...compress import CompressionManager
from ...infra.logging import LogWrapper
from ...infra.session_recording import SessionRecorder
from ...tools import ToolCall, ToolExecutionContext, ToolExecutor, ToolRegistry
from ..base_runner import BaseAgentRunner
from ..message_visibility import COMPRESS_TOOL_NAME, MessageVisibilityManager
from .message_conversion import (
    internal_history_to_langchain_messages,
    langchain_ai_message_to_internal_assistant_message,
)
from .tool_bridge import wrap_registry_tools


TRACE_TAG = "trace"


class LangGraphTurnState(TypedDict):
    """State carried by the LangGraph turn graph."""

    message_state: MessageState
    latest_ai_message: LangChainAIMessage | None
    latest_assistant_message: AssistantMessage | None
    latest_assistant_message_index: int | None
    model_rounds: int
    final_text: str | None


class LangGraphAgentRunner(BaseAgentRunner):
    """Run one user turn through a minimal LangGraph model/tools graph."""

    def __init__(
        self,
        chat_model: Any,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        max_tool_rounds: int,
        compression_manager: CompressionManager | None = None,
        message_visibility_manager: MessageVisibilityManager | None = None,
    ) -> None:
        self._chat_model = chat_model.bind_tools(wrap_registry_tools(tool_registry))
        self._tool_executor = tool_executor
        self._max_tool_rounds = max_tool_rounds
        self._compression_manager = compression_manager
        self._message_visibility_manager = message_visibility_manager or MessageVisibilityManager()
        self._graph = self._build_graph()

    def run_turn(self, message_state: MessageState) -> str:
        """Run the LangGraph-backed agent loop for one user turn."""
        result = self._graph.invoke(
            LangGraphTurnState(
                message_state=message_state,
                latest_ai_message=None,
                latest_assistant_message=None,
                latest_assistant_message_index=None,
                model_rounds=0,
                final_text=None,
            )
        )
        final_text = result.get("final_text")
        if not isinstance(final_text, str):
            raise RuntimeError("LLM 最终响应缺少文本 content")
        return final_text

    def _build_graph(self):
        graph = StateGraph(LangGraphTurnState)
        graph.add_node("model", self._model_node)
        graph.add_node("tools", self._tools_node)
        graph.set_entry_point("model")
        graph.add_conditional_edges(
            "model",
            self._route_after_model,
            {
                "tools": "tools",
                "end": END,
            },
        )
        graph.add_edge("tools", "model")
        return graph.compile()

    def _model_node(self, state: LangGraphTurnState) -> dict[str, Any]:
        model_rounds = state["model_rounds"]
        if model_rounds >= self._max_tool_rounds:
            raise RuntimeError(f"工具调用轮数超过上限：{self._max_tool_rounds}")

        round_number = model_rounds + 1
        LogWrapper.debug(TRACE_TAG, f"round={round_number}")

        message_state = state["message_state"]
        if self._compression_manager is not None:
            self._compression_manager.compress_if_needed(message_state)

        ai_message = self._chat_model.invoke(
            internal_history_to_langchain_messages(message_state.model_visible_history())
        )
        if not isinstance(ai_message, LangChainAIMessage):
            raise RuntimeError("LangGraph model response is not an AIMessage")

        assistant_message = langchain_ai_message_to_internal_assistant_message(ai_message)
        self._validate_tool_call_batch(assistant_message)
        assistant_message_index = len(message_state.history)
        message_state.history.append(assistant_message)
        SessionRecorder.record_session(
            "AssistantMessage",
            {
                "content": assistant_message.content,
                "tool_calls": [asdict(tool_call) for tool_call in assistant_message.tool_calls],
            },
            "json",
        )

        final_text = None
        if not assistant_message.tool_calls:
            if not isinstance(assistant_message.content, str):
                raise RuntimeError("LLM 最终响应缺少文本 content")
            final_text = assistant_message.content

        return {
            "latest_ai_message": ai_message,
            "latest_assistant_message": assistant_message,
            "latest_assistant_message_index": assistant_message_index,
            "model_rounds": round_number,
            "final_text": final_text,
        }

    def _route_after_model(self, state: LangGraphTurnState) -> Literal["tools", "end"]:
        assistant_message = state["latest_assistant_message"]
        if assistant_message is None:
            raise RuntimeError("LangGraph runner missing assistant message")
        if assistant_message.tool_calls:
            return "tools"
        return "end"

    def _tools_node(self, state: LangGraphTurnState) -> dict[str, Any]:
        message_state = state["message_state"]
        assistant_message = state["latest_assistant_message"]
        assistant_message_index = state["latest_assistant_message_index"]
        if assistant_message is None or assistant_message_index is None:
            raise RuntimeError("LangGraph runner missing tool call source")

        tool_call_count = len(assistant_message.tool_calls)
        tool_message_indexes: list[int] = []
        for tool_call_index, tool_call in enumerate(assistant_message.tool_calls, start=1):
            LogWrapper.debug(
                TRACE_TAG,
                self._tool_call_log_line(
                    round_number=state["model_rounds"],
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
            tool_message_index = len(message_state.history)
            message_state.history.append(
                ToolMessage(
                    tool_call_id=tool_call.id,
                    content=tool_content,
                )
            )
            tool_message_indexes.append(tool_message_index)
            SessionRecorder.record_session(f"ToolResult {tool_call.name}", tool_content, "text")

        self._message_visibility_manager.apply_after_tool_batch(
            message_state=message_state,
            assistant_message_index=assistant_message_index,
            tool_message_indexes=tool_message_indexes,
        )
        return {}

    @staticmethod
    def _validate_tool_call_batch(assistant_message: AssistantMessage) -> None:
        tool_names = {tool_call.name for tool_call in assistant_message.tool_calls}
        if COMPRESS_TOOL_NAME in tool_names and tool_names != {COMPRESS_TOOL_NAME}:
            raise RuntimeError("compress_tool cannot be mixed with other tool calls")

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
