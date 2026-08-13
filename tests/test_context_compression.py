from __future__ import annotations

import unittest

from ai_job.compress import (
    CompressionManager,
    CompressionPlan,
    CompressionTrigger,
    MessageRange,
    build_compression_plan,
    check_compression_trigger,
)
from ai_job.communication import (
    AssistantMessage,
    MessageState,
    SummaryMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ai_job.tools import ToolCall


def one_token(_message):
    return 1


class ContextCompressionTriggerTest(unittest.TestCase):
    def test_check_trigger_compresses_only_when_tokens_exceed_threshold(self):
        active_context = [
            SystemMessage(content="sys"),
            UserMessage(content="hello"),
            AssistantMessage(content="hi"),
        ]

        self.assertEqual(
            check_compression_trigger(
                active_context=active_context,
                context_window=5,
                reserve_tokens=2,
                token_counter=one_token,
            ),
            CompressionTrigger(
                compression_threshold=3,
                active_context_tokens=3,
                should_compress=False,
            ),
        )
        self.assertEqual(
            check_compression_trigger(
                active_context=active_context,
                context_window=4,
                reserve_tokens=2,
                token_counter=one_token,
            ),
            CompressionTrigger(
                compression_threshold=2,
                active_context_tokens=3,
                should_compress=True,
            ),
        )

    def test_check_trigger_rejects_invalid_window_settings(self):
        active_context = [UserMessage(content="hello")]

        with self.assertRaisesRegex(ValueError, "context_window must be positive"):
            check_compression_trigger(active_context, 0, 0, one_token)
        with self.assertRaisesRegex(ValueError, "reserve_tokens must be non-negative"):
            check_compression_trigger(active_context, 10, -1, one_token)
        with self.assertRaisesRegex(ValueError, "reserve_tokens must be smaller"):
            check_compression_trigger(active_context, 10, 10, one_token)

    def test_check_trigger_rejects_negative_token_count(self):
        with self.assertRaisesRegex(ValueError, "token_counter must not return negative"):
            check_compression_trigger(
                active_context=[UserMessage(content="hello")],
                context_window=10,
                reserve_tokens=1,
                token_counter=lambda _message: -1,
            )


class ContextCompressionPlanTest(unittest.TestCase):
    def test_build_plan_matches_split_turn_example(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="u1"),
            AssistantMessage(content="a2"),
            UserMessage(content="u3"),
            AssistantMessage(content="a4"),
            ToolMessage(tool_call_id="call-4", content="t5"),
            AssistantMessage(content="a6"),
            ToolMessage(tool_call_id="call-6", content="t7"),
            AssistantMessage(content="a8"),
            ToolMessage(tool_call_id="call-8", content="t9"),
        ]

        plan = build_compression_plan(
            history=history,
            context_start_index=1,
            keep_recent_tokens=2,
            token_counter=one_token,
        )

        self.assertEqual(
            plan,
            CompressionPlan(
                complete_range=MessageRange(1, 3),
                split_range=MessageRange(3, 8),
                keep_range=MessageRange(8, 10),
            ),
        )

    def test_build_plan_keeps_tool_message_with_matching_assistant(self):
        matching_tool_call = ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"})
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="old"),
            AssistantMessage(content=None, tool_calls=[matching_tool_call]),
            ToolMessage(tool_call_id="call-1", content="file content"),
        ]

        plan = build_compression_plan(
            history=history,
            context_start_index=1,
            keep_recent_tokens=1,
            token_counter=one_token,
        )

        self.assertEqual(plan.complete_range, MessageRange(1, 1))
        self.assertEqual(plan.split_range, MessageRange(1, 2))
        self.assertEqual(plan.keep_range, MessageRange(2, 4))

    def test_build_plan_does_not_create_split_when_keep_starts_at_user_message(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="active"),
            AssistantMessage(content="active answer"),
        ]

        plan = build_compression_plan(
            history=history,
            context_start_index=1,
            keep_recent_tokens=2,
            token_counter=one_token,
        )

        self.assertEqual(plan.complete_range, MessageRange(1, 3))
        self.assertIsNone(plan.split_range)
        self.assertEqual(plan.keep_range, MessageRange(3, 5))

    def test_build_plan_uses_context_start_index_as_compressible_start(self):
        history = [
            SystemMessage(content="sys"),
            SummaryMessage(complete_turn_summary="summary"),
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="active"),
            AssistantMessage(content="active answer"),
        ]

        plan = build_compression_plan(
            history=history,
            context_start_index=1,
            keep_recent_tokens=2,
            token_counter=one_token,
        )

        self.assertEqual(plan.complete_range, MessageRange(1, 4))
        self.assertIsNone(plan.split_range)
        self.assertEqual(plan.keep_range, MessageRange(4, 6))

    def test_build_plan_rejects_when_keep_recent_tokens_leave_no_summary_range(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="active"),
            AssistantMessage(content="active answer"),
        ]

        with self.assertRaisesRegex(ValueError, "leaves no messages to summarize"):
            build_compression_plan(
                history=history,
                context_start_index=1,
                keep_recent_tokens=99,
                token_counter=one_token,
            )


class CompressionManagerTest(unittest.TestCase):
    def test_compress_if_needed_does_not_call_summarizer_below_threshold(self):
        calls = []
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="hello"),
        ]
        manager = CompressionManager(
            context_window=10,
            reserve_tokens=2,
            keep_recent_tokens=1,
            token_counter=one_token,
            summarizer=lambda plan, current_history: calls.append((plan, current_history)),
        )

        manager.compress_if_needed(MessageState(history=history))

        self.assertEqual(calls, [])
        self.assertEqual(
            history,
            [
                SystemMessage(content="sys"),
                UserMessage(content="hello"),
            ],
        )

    def test_compress_if_needed_appends_compressed_context_and_moves_context_start(self):
        original_history = [
            SystemMessage(content="sys"),
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="active"),
            AssistantMessage(content="active answer"),
        ]
        history = list(original_history)
        summary = SummaryMessage(complete_turn_summary="compressed")
        received = []

        def summarize(plan, current_message_state):
            received.append((plan, list(current_message_state.history)))
            return summary

        manager = CompressionManager(
            context_window=4,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=one_token,
            summarizer=summarize,
        )

        message_state = MessageState(history=history)

        manager.compress_if_needed(message_state)

        self.assertEqual(history[: len(original_history)], original_history)
        self.assertEqual(history[len(original_history)], summary)
        self.assertEqual(history[len(original_history) + 1 :], original_history[3:])
        self.assertEqual(message_state.context_start_index, len(original_history))
        self.assertEqual(received[0][0].complete_range, MessageRange(1, 3))
        self.assertEqual(received[0][0].keep_range, MessageRange(3, 5))
        self.assertEqual(received[0][1], original_history)

    def test_compress_if_needed_ignores_hidden_messages_when_checking_threshold(self):
        calls = []
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="visible"),
            AssistantMessage(content="hidden", visible_to_model=False),
        ]
        manager = CompressionManager(
            context_window=3,
            reserve_tokens=1,
            keep_recent_tokens=1,
            token_counter=one_token,
            summarizer=lambda plan, message_state: calls.append((plan, message_state)),
        )

        manager.compress_if_needed(MessageState(history=history))

        self.assertEqual(calls, [])
        self.assertEqual(
            history,
            [
                SystemMessage(content="sys"),
                UserMessage(content="visible"),
                AssistantMessage(content="hidden", visible_to_model=False),
            ],
        )

    def test_compress_if_needed_propagates_summarizer_failure(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="active"),
            AssistantMessage(content="active answer"),
        ]
        original_history = list(history)

        def fail(_plan, _history):
            raise RuntimeError("summary failed")

        manager = CompressionManager(
            context_window=4,
            reserve_tokens=1,
            keep_recent_tokens=2,
            token_counter=one_token,
            summarizer=fail,
        )

        with self.assertRaisesRegex(RuntimeError, "summary failed"):
            manager.compress_if_needed(MessageState(history=history))

        self.assertEqual(history, original_history)


if __name__ == "__main__":
    unittest.main()
