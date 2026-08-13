"""Stateful context compression coordinator."""

from __future__ import annotations

from collections.abc import Callable

from ..communication import MessageHistory, MessageState, SummaryMessage, SystemMessage
from .context_compression import (
    CompressionPlan,
    TokenCounter,
    build_compression_plan,
    check_compression_trigger,
)


Summarizer = Callable[[CompressionPlan, MessageHistory], SummaryMessage]


class CompressionManager:
    """Check context budget and rewrite history with a generated summary when needed."""

    def __init__(
        self,
        context_window: int,
        reserve_tokens: int,
        keep_recent_tokens: int,
        token_counter: TokenCounter,
        summarizer: Summarizer,
    ) -> None:
        self._context_window = context_window
        self._reserve_tokens = reserve_tokens
        self._keep_recent_tokens = keep_recent_tokens
        self._token_counter = token_counter
        self._summarizer = summarizer

    def compress_if_needed(self, message_state: MessageState) -> None:
        """Mutate history in-place when the active context exceeds the compression threshold."""
        history = message_state.history
        active_context = self._active_context(message_state)
        trigger = check_compression_trigger(
            active_context=active_context,
            context_window=self._context_window,
            reserve_tokens=self._reserve_tokens,
            token_counter=self._token_counter,
        )
        if not trigger.should_compress:
            return

        plan = build_compression_plan(
            history=history,
            context_start_index=message_state.context_start_index,
            keep_recent_tokens=self._keep_recent_tokens,
            token_counter=self._token_counter,
        )
        summary = self._summarizer(plan, history)
        self._compressed_history(message_state, plan, summary)

    def _active_context(self, message_state: MessageState) -> MessageHistory:
        history = message_state.history
        if not history:
            return []

        active_messages = history[message_state.context_start_index :]
        if message_state.context_start_index > 0 and _has_system_message(history):
            return [history[0], *active_messages]
        return active_messages

    @staticmethod
    def _compressed_history(
        message_state: MessageState,
        plan: CompressionPlan,
        summary: SummaryMessage,
    ) -> None:
        history = message_state.history
        summary_index = len(history)
        recent_messages = history[plan.keep_range.start : plan.keep_range.end]
        message_state.history.extend(
            [
                summary,
                *recent_messages,
            ]
        )
        message_state.context_start_index = summary_index


def _has_system_message(history: MessageHistory) -> bool:
    return bool(history) and isinstance(history[0], SystemMessage)
