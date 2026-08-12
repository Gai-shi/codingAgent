"""Pure planning rules for context compression."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..communication import (
    AssistantMessage,
    Message,
    MessageHistory,
    SummaryMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


TokenCounter = Callable[[Message], int]


@dataclass(frozen=True)
class MessageRange:
    """Half-open message range matching Python slicing semantics."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("MessageRange.start must be non-negative")
        if self.end < self.start:
            raise ValueError("MessageRange.end must be greater than or equal to start")


@dataclass(frozen=True)
class CompressionPlan:
    """Message ranges to summarize and preserve during one compression pass."""

    complete_range: MessageRange
    split_range: MessageRange | None
    keep_range: MessageRange


@dataclass(frozen=True)
class CompressionTrigger:
    """Token threshold check result for the current active context."""

    compression_threshold: int
    active_context_tokens: int
    should_compress: bool


def check_compression_trigger(
    active_context: MessageHistory,
    context_window: int,
    reserve_tokens: int,
    token_counter: TokenCounter,
) -> CompressionTrigger:
    """Check whether the current active context crosses the compression threshold."""
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if reserve_tokens < 0:
        raise ValueError("reserve_tokens must be non-negative")
    if reserve_tokens >= context_window:
        raise ValueError("reserve_tokens must be smaller than context_window")

    compression_threshold = context_window - reserve_tokens
    active_context_tokens = _count_tokens(active_context, token_counter)
    return CompressionTrigger(
        compression_threshold=compression_threshold,
        active_context_tokens=active_context_tokens,
        should_compress=active_context_tokens > compression_threshold,
    )


def build_compression_plan(
    history: MessageHistory,
    context_start_index: int,
    keep_recent_tokens: int,
    token_counter: TokenCounter,
) -> CompressionPlan:
    """Build a compression plan after the caller has already decided to compress."""
    if context_start_index < 0 or context_start_index > len(history):
        raise ValueError("context_start_index is out of range")
    if keep_recent_tokens <= 0:
        raise ValueError("keep_recent_tokens must be positive")

    compressible_start = _compressible_start(history, context_start_index)
    if compressible_start >= len(history):
        raise ValueError("history has no active messages to compress")

    raw_keep_start = _raw_keep_start(history, compressible_start, keep_recent_tokens, token_counter)
    keep_start = _safe_keep_start(history, compressible_start, raw_keep_start)
    if keep_start <= compressible_start:
        raise ValueError("keep_recent_tokens leaves no messages to summarize")

    split_start = _split_start(history, compressible_start, keep_start)
    complete_end = split_start if split_start is not None else keep_start
    split_range = MessageRange(split_start, keep_start) if split_start is not None else None

    return CompressionPlan(
        complete_range=MessageRange(compressible_start, complete_end),
        split_range=split_range,
        keep_range=MessageRange(keep_start, len(history)),
    )


def _compressible_start(history: MessageHistory, context_start_index: int) -> int:
    if context_start_index == 0 and history and isinstance(history[0], SystemMessage):
        return 1
    return context_start_index


def _count_tokens(messages: MessageHistory, token_counter: TokenCounter) -> int:
    token_total = 0
    for message in messages:
        token_count = token_counter(message)
        if token_count < 0:
            raise ValueError("token_counter must not return negative counts")
        token_total += token_count
    return token_total


def _raw_keep_start(
    history: MessageHistory,
    compressible_start: int,
    keep_recent_tokens: int,
    token_counter: TokenCounter,
) -> int:
    token_total = 0
    for index in range(len(history) - 1, compressible_start - 1, -1):
        token_count = token_counter(history[index])
        if token_count < 0:
            raise ValueError("token_counter must not return negative counts")
        token_total += token_count
        if token_total >= keep_recent_tokens:
            return index
    return compressible_start


def _safe_keep_start(history: MessageHistory, compressible_start: int, raw_keep_start: int) -> int:
    raw_message = history[raw_keep_start]
    if isinstance(raw_message, ToolMessage):
        return _matching_assistant_index(history, compressible_start, raw_keep_start, raw_message)
    if isinstance(raw_message, (UserMessage, SummaryMessage, AssistantMessage)):
        return raw_keep_start
    raise TypeError(f"unsupported keep-start message type: {type(raw_message).__name__}")


def _matching_assistant_index(
    history: MessageHistory,
    compressible_start: int,
    tool_index: int,
    tool_message: ToolMessage,
) -> int:
    for index in range(tool_index - 1, compressible_start - 1, -1):
        message = history[index]
        if not isinstance(message, AssistantMessage):
            continue
        tool_call_ids = {tool_call.id for tool_call in message.tool_calls}
        if tool_message.tool_call_id in tool_call_ids:
            return index
    raise ValueError("ToolMessage cannot be kept without its matching AssistantMessage")


def _split_start(history: MessageHistory, compressible_start: int, keep_start: int) -> int | None:
    if not isinstance(history[keep_start], AssistantMessage):
        return None

    for index in range(keep_start - 1, compressible_start - 1, -1):
        if isinstance(history[index], UserMessage):
            return index
    return None
