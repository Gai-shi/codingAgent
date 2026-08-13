from __future__ import annotations

import json
import unittest

from ai_job.composition import build_summary_messages, parse_summary_message
from ai_job.compress import CompressionPlan, MessageRange
from ai_job.communication import (
    AssistantMessage,
    MessageState,
    SummaryMessage,
    SystemMessage,
    UserMessage,
)


class CompressionFactoryTest(unittest.TestCase):
    def test_build_summary_messages_uses_plan_ranges(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="old"),
            AssistantMessage(content="old answer"),
            UserMessage(content="split"),
            AssistantMessage(content="kept"),
        ]
        plan = CompressionPlan(
            complete_range=MessageRange(1, 3),
            split_range=MessageRange(3, 4),
            keep_range=MessageRange(4, 5),
        )

        summary_messages = build_summary_messages(plan, MessageState(history=history))

        self.assertEqual(len(summary_messages), 1)
        content = summary_messages[0].content
        self.assertIn("complete_messages", content)
        self.assertIn("split_messages", content)
        self.assertIn("old answer", content)
        self.assertIn("split", content)
        self.assertNotIn("kept", content)

    def test_build_summary_messages_filters_hidden_messages(self):
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="visible old"),
            AssistantMessage(content="hidden compress call", visible_to_model=False),
            UserMessage(content="visible split"),
            AssistantMessage(content="kept"),
        ]
        plan = CompressionPlan(
            complete_range=MessageRange(1, 3),
            split_range=MessageRange(3, 4),
            keep_range=MessageRange(4, 5),
        )

        summary_messages = build_summary_messages(plan, MessageState(history=history))

        content = summary_messages[0].content
        self.assertIn("visible old", content)
        self.assertIn("visible split", content)
        self.assertNotIn("hidden compress call", content)
        self.assertNotIn("kept", content)

    def test_parse_summary_message_parses_json_summary(self):
        assistant_message = AssistantMessage(
            content=json.dumps(
                {
                    "complete_turn_summary": "complete",
                    "split_turn_summary": None,
                }
            )
        )

        self.assertEqual(
            parse_summary_message(assistant_message),
            SummaryMessage(complete_turn_summary="complete"),
        )

    def test_parse_summary_message_rejects_invalid_content(self):
        with self.assertRaisesRegex(RuntimeError, "不是合法 JSON"):
            parse_summary_message(AssistantMessage(content="not json"))
        with self.assertRaisesRegex(RuntimeError, "缺少 complete_turn_summary"):
            parse_summary_message(AssistantMessage(content='{"complete_turn_summary": ""}'))


if __name__ == "__main__":
    unittest.main()
