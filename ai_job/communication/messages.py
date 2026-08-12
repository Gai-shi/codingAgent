"""Agent-internal conversation message types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Union

from ..tools.tool_call import ToolCall


@dataclass(frozen=True)
class SystemMessage:
    """Instruction message that establishes the agent's behavior."""

    content: str
    visible_to_model: bool = True


@dataclass(frozen=True)
class UserMessage:
    """Message created from one user input that should enter model context."""

    content: str
    visible_to_model: bool = True


@dataclass(frozen=True)
class SummaryMessage:
    """Compressed conversation context produced by the agent."""

    complete_turn_summary: str
    split_turn_summary: str | None = None
    visible_to_model: bool = True


@dataclass(frozen=True)
class AssistantMessage:
    """Provider-normalized assistant message returned by a chat model."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    visible_to_model: bool = True


@dataclass(frozen=True)
class ToolMessage:
    """Tool execution text that should be sent back to the model."""

    tool_call_id: str
    content: str
    compressions: list[str] = field(default_factory=list)
    visible_to_model: bool = True


Message = Union[SystemMessage, UserMessage, SummaryMessage, AssistantMessage, ToolMessage]
MessageHistory = list[Message]


def message_to_debug_dict(message: Message) -> dict[str, Any]:
    """Convert one internal message into a JSON-serializable debug dictionary."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, SummaryMessage):
        return {
            "role": "summary",
            "complete_turn_summary": message.complete_turn_summary,
            "split_turn_summary": message.split_turn_summary,
        }
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [asdict(tool_call) for tool_call in message.tool_calls],
        }
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }

    raise TypeError(f"unknown message type: {type(message).__name__}")


def message_history_to_debug_dicts(history: MessageHistory) -> list[dict[str, Any]]:
    """Convert internal message history into JSON-serializable debug dictionaries."""
    return [message_to_debug_dict(message) for message in history]
