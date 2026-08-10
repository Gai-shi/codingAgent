from __future__ import annotations

import unittest

from ai_job.agent.token_counting import estimate_message_tokens, estimate_text_tokens
from ai_job.communication import AssistantMessage, SummaryMessage, SystemMessage, ToolMessage, UserMessage
from ai_job.tools import ToolCall


class TokenCountingTest(unittest.TestCase):
    def test_estimate_text_tokens_uses_chars_per_four_rounded_up(self):
        self.assertEqual(estimate_text_tokens(""), 0)
        self.assertEqual(estimate_text_tokens("a"), 1)
        self.assertEqual(estimate_text_tokens("abcd"), 1)
        self.assertEqual(estimate_text_tokens("abcde"), 2)

    def test_estimate_message_tokens_counts_basic_message_content(self):
        self.assertEqual(estimate_message_tokens(SystemMessage(content="abcd")), 1)
        self.assertEqual(estimate_message_tokens(UserMessage(content="abcde")), 2)
        self.assertEqual(estimate_message_tokens(ToolMessage(tool_call_id="call-1", content="abcdefghi")), 3)

    def test_estimate_message_tokens_counts_summary_fields(self):
        self.assertEqual(
            estimate_message_tokens(
                SummaryMessage(
                    complete_turn_summary="abcd",
                    split_turn_summary="efgh",
                )
            ),
            3,
        )
        self.assertEqual(
            estimate_message_tokens(SummaryMessage(complete_turn_summary="abcd")),
            1,
        )

    def test_estimate_message_tokens_counts_assistant_content_and_tool_calls(self):
        message = AssistantMessage(
            content="answer",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="read_file",
                    arguments={"path": "a.py", "limit": 10},
                )
            ],
        )

        self.assertGreater(estimate_message_tokens(message), estimate_text_tokens("answer"))

    def test_estimate_message_tokens_rejects_unknown_message_type(self):
        with self.assertRaisesRegex(TypeError, "unknown message type"):
            estimate_message_tokens(object())


if __name__ == "__main__":
    unittest.main()
