"""Lightweight token estimation for agent messages."""

from __future__ import annotations

import json
import math
from typing import Any

from ..communication import (
    AssistantMessage,
    Message,
    SummaryMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    tool_message_visible_content,
)


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens from plain text using a simple chars/4 heuristic."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_message_tokens(message: Message) -> int:
    """Estimate token count for one internal message."""
    if isinstance(message, SystemMessage):
        return estimate_text_tokens(message.content)
    if isinstance(message, UserMessage):
        return estimate_text_tokens(message.content)
    if isinstance(message, SummaryMessage):
        return estimate_text_tokens(_summary_text(message))
    if isinstance(message, AssistantMessage):
        return estimate_text_tokens(_assistant_text(message))
    if isinstance(message, ToolMessage):
        return estimate_text_tokens(tool_message_visible_content(message))

    raise TypeError(f"unknown message type: {type(message).__name__}")


def _summary_text(message: SummaryMessage) -> str:
    parts = [message.complete_turn_summary]
    if message.split_turn_summary is not None:
        parts.append(message.split_turn_summary)
    return "\n".join(parts)


def _assistant_text(message: AssistantMessage) -> str:
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    for tool_call in message.tool_calls:
        parts.append(tool_call.name)
        parts.append(_stable_json(tool_call.arguments))
    return "\n".join(parts)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
