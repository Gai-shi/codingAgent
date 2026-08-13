"""Agent-internal conversation message types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Union

from ..tools.tool_call import ToolCall


@dataclass
class SystemMessage:
    """Instruction message that establishes the agent's behavior."""

    content: str
    visible_to_model: bool = True


@dataclass
class UserMessage:
    """Message created from one user input that should enter model context."""

    content: str
    visible_to_model: bool = True


@dataclass
class SummaryMessage:
    """Compressed conversation context produced by the agent."""

    complete_turn_summary: str
    split_turn_summary: str | None = None
    visible_to_model: bool = True


@dataclass
class AssistantMessage:
    """Provider-normalized assistant message returned by a chat model."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    visible_to_model: bool = True


@dataclass
class ToolMessage:
    """Tool execution text that should be sent back to the model."""

    tool_call_id: str
    content: str
    compressions: list[str] = field(default_factory=list)
    visible_to_model: bool = True


Message = Union[SystemMessage, UserMessage, SummaryMessage, AssistantMessage, ToolMessage]
MessageHistory = list[Message]


@dataclass
class MessageState:
    """Mutable conversation state shared by agent loop and tools."""

    history: MessageHistory
    context_start_index: int = 1

    def model_visible_history(self) -> MessageHistory:
        """Return the single source of truth for messages sent to the model."""
        self._validate_context_start_index()
        candidate_messages = [self.history[0], *self.history[self.context_start_index :]]
        return self._visible_messages(candidate_messages)

    def visible_history_range(self, start: int, end: int) -> MessageHistory:
        """Return visible messages inside one original-history half-open range."""
        if start < 0 or end < start or end > len(self.history):
            raise ValueError("message range is out of range")
        return self._visible_messages(self.history[start:end])

    def _validate_context_start_index(self) -> None:
        if not self.history:
            raise ValueError("history must contain a system message")
        if not isinstance(self.history[0], SystemMessage):
            raise ValueError("history[0] must be a SystemMessage")
        if self.context_start_index < 1:
            raise ValueError("context_start_index must be at least 1")
        if self.context_start_index > len(self.history):
            raise ValueError("context_start_index is out of range")

    @staticmethod
    def _visible_messages(messages: MessageHistory) -> MessageHistory:
        return [message for message in messages if message.visible_to_model]


def tool_message_visible_content(message: ToolMessage) -> str:
    """Return the ToolMessage content that should enter model context."""
    if message.compressions:
        return message.compressions[-1]
    return message.content


def message_to_debug_dict(message: Message) -> dict[str, Any]:
    """Convert one internal message into a JSON-serializable debug dictionary."""
    return _message_to_dict(message, use_model_visible_content=False)


def message_to_model_visible_debug_dict(message: Message) -> dict[str, Any]:
    """Convert one model-visible message using the exact content the model should see."""
    return _message_to_dict(message, use_model_visible_content=True)


def _message_to_dict(message: Message, *, use_model_visible_content: bool) -> dict[str, Any]:
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
            "content": tool_message_visible_content(message)
            if use_model_visible_content
            else message.content,
        }

    raise TypeError(f"unknown message type: {type(message).__name__}")


def message_history_to_debug_dicts(history: MessageHistory) -> list[dict[str, Any]]:
    """Convert internal message history into JSON-serializable debug dictionaries."""
    return [message_to_debug_dict(message) for message in history]


def message_history_to_model_visible_debug_dicts(history: MessageHistory) -> list[dict[str, Any]]:
    """Convert model-visible messages into JSON-serializable dictionaries for model prompts."""
    return [message_to_model_visible_debug_dict(message) for message in history]
