"""Convert ai_job messages to and from LangChain message objects."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage as LangChainAIMessage
from langchain_core.messages import BaseMessage as LangChainBaseMessage
from langchain_core.messages import HumanMessage as LangChainHumanMessage
from langchain_core.messages import SystemMessage as LangChainSystemMessage
from langchain_core.messages import ToolMessage as LangChainToolMessage

from ...communication import (
    AssistantMessage,
    Message,
    MessageHistory,
    SummaryMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    tool_message_visible_content,
)
from ...tools import ToolCall


def internal_history_to_langchain_messages(history: MessageHistory) -> list[LangChainBaseMessage]:
    """Convert internal message history into LangChain message history."""
    return [internal_message_to_langchain_message(message) for message in history]


def internal_message_to_langchain_message(message: Message) -> LangChainBaseMessage:
    """Convert one internal message into the equivalent LangChain message."""
    if isinstance(message, SystemMessage):
        return LangChainSystemMessage(content=message.content)
    if isinstance(message, UserMessage):
        return LangChainHumanMessage(content=message.content)
    if isinstance(message, SummaryMessage):
        return LangChainHumanMessage(content=_render_summary_content(message))
    if isinstance(message, AssistantMessage):
        return LangChainAIMessage(
            content=message.content or "",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ],
        )
    if isinstance(message, ToolMessage):
        return LangChainToolMessage(
            content=tool_message_visible_content(message),
            tool_call_id=message.tool_call_id,
        )

    raise TypeError(f"unknown message type: {type(message).__name__}")


def langchain_ai_message_to_internal_assistant_message(
    message: LangChainAIMessage,
) -> AssistantMessage:
    """Convert a LangChain AIMessage into an internal AssistantMessage."""
    if not isinstance(message.content, str):
        raise RuntimeError("LangGraph assistant content must be text")

    return AssistantMessage(
        content=message.content,
        tool_calls=[
            _langchain_tool_call_to_internal_tool_call(tool_call)
            for tool_call in message.tool_calls
        ],
    )


def _langchain_tool_call_to_internal_tool_call(tool_call: dict[str, Any]) -> ToolCall:
    tool_call_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    tool_arguments = tool_call.get("args")

    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise RuntimeError("LangGraph tool call is missing id")
    if not isinstance(tool_name, str) or not tool_name:
        raise RuntimeError("LangGraph tool call is missing name")
    if not isinstance(tool_arguments, dict):
        raise RuntimeError("LangGraph tool call args must be an object")

    return ToolCall(id=tool_call_id, name=tool_name, arguments=tool_arguments)


def _render_summary_content(message: SummaryMessage) -> str:
    parts = [
        "以下是之前对话的压缩摘要。",
        "",
        "完整回合摘要：",
        message.complete_turn_summary,
    ]
    if message.split_turn_summary is not None:
        parts.extend(
            [
                "",
                "当前未完整保留回合的前半段摘要：",
                message.split_turn_summary,
            ]
        )
    return "\n".join(parts)
