"""Stateful context compression coordinator."""

from __future__ import annotations

from collections.abc import Callable

from ..communication import MessageState, SummaryMessage
from .context_compression import (
    CompressionPlan,
    TokenCounter,
    build_compression_plan,
    check_compression_trigger,
)


Summarizer = Callable[[CompressionPlan, MessageState], SummaryMessage]


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
        trigger = check_compression_trigger(
            active_context=message_state.model_visible_history(),
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
        summary = self._summarizer(plan, message_state)
        self._compressed_history(message_state, plan, summary)

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
