from __future__ import annotations

import unittest

from ai_job.agent.context_compression import (
    CompressionPlan,
    CompressionTrigger,
    MessageRange,
    build_compression_plan,
    check_compression_trigger,
)
from ai_job.communication import AssistantMessage, SummaryMessage, SystemMessage, ToolMessage, UserMessage
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
            context_start_index=0,
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
            context_start_index=0,
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
            context_start_index=0,
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
                context_start_index=0,
                keep_recent_tokens=99,
                token_counter=one_token,
            )


if __name__ == "__main__":
    unittest.main()
