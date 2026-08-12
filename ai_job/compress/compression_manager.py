"""Stateful context compression coordinator."""

from __future__ import annotations

from collections.abc import Callable

from ..communication import MessageHistory, SummaryMessage, SystemMessage
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
        context_start_index: int = 0,
    ) -> None:
        self._context_window = context_window
        self._reserve_tokens = reserve_tokens
        self._keep_recent_tokens = keep_recent_tokens
        self._token_counter = token_counter
        self._summarizer = summarizer
        self._context_start_index = context_start_index

    def compress_if_needed(self, history: MessageHistory) -> None:
        """Mutate history in-place when the active context exceeds the compression threshold."""
        active_context = self._active_context(history)
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
            context_start_index=self._context_start_index,
            keep_recent_tokens=self._keep_recent_tokens,
            token_counter=self._token_counter,
        )
        summary = self._summarizer(plan, history)
        history[:] = self._compressed_history(history, plan, summary)

    def _active_context(self, history: MessageHistory) -> MessageHistory:
        if not history:
            return []

        active_messages = history[self._context_start_index :]
        if self._context_start_index > 0 and isinstance(history[0], SystemMessage):
            return [history[0], *active_messages]
        return active_messages

    @staticmethod
    def _compressed_history(
        history: MessageHistory,
        plan: CompressionPlan,
        summary: SummaryMessage,
    ) -> MessageHistory:
        prefix = [history[0]] if history and isinstance(history[0], SystemMessage) else []
        return [
            *prefix,
            summary,
            *history[plan.keep_range.start : plan.keep_range.end],
        ]
