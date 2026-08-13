from __future__ import annotations

import unittest

from ai_job.communication import AssistantMessage, MessageState, SystemMessage, ToolMessage, UserMessage
from ai_job.tools import BaseTool, CompressTool, ToolCall, ToolExecutionContext, ToolExecutor, ToolRegistry


class SuccessfulTool(BaseTool):
    name = "success"
    description = "Successful test tool."

    def _run(self, arguments):
        return "ok"


class FailingTool(BaseTool):
    name = "failure"
    description = "Failing test tool."

    def _run(self, arguments):
        raise RuntimeError("boom")


class NonStringTool(BaseTool):
    name = "non_string"
    description = "Tool returning an invalid value."

    def _run(self, arguments):
        return {"not": "a string"}


class ToolContractsTest(unittest.TestCase):
    def test_tool_registry_maps_names_and_rejects_duplicates(self):
        registry = ToolRegistry([SuccessfulTool()])

        self.assertEqual(registry.names(), ["success"])
        self.assertIsInstance(registry.get("success"), SuccessfulTool)
        self.assertIsNone(registry.get("missing"))

        with self.assertRaisesRegex(ValueError, "duplicate tool name"):
            ToolRegistry([SuccessfulTool(), SuccessfulTool()])

    def test_tool_registry_rejects_empty_tool_name(self):
        class EmptyNameTool(BaseTool):
            name = ""

        with self.assertRaisesRegex(ValueError, "tool name must not be empty"):
            ToolRegistry([EmptyNameTool()])

    def test_tool_executor_returns_unknown_tool_error(self):
        executor = ToolExecutor(ToolRegistry([]))

        result = executor.execute(ToolCall(id="call-1", name="missing", arguments={}))

        self.assertEqual(result, "Error: unknown tool: missing")

    def test_base_tool_wraps_runtime_failures_as_error_text(self):
        result = FailingTool().execute({})

        self.assertEqual(result, "Error: boom")

    def test_base_tool_wraps_non_string_return_as_error_text(self):
        result = NonStringTool().execute({})

        self.assertIn("Error: tool non_string returned non-string value", result)

    def test_base_tool_rejects_non_dict_arguments(self):
        result = SuccessfulTool().execute("not a dict")

        self.assertEqual(result, "Error: invalid tool arguments: expected a JSON object")

    def test_compress_tool_uses_model_visible_history(self):
        old_tool_message = ToolMessage(tool_call_id="call-1", content="old result")
        active_tool_message = ToolMessage(tool_call_id="call-1", content="active result")
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "old.txt"})
                    ],
                ),
                old_tool_message,
                UserMessage(content="active"),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "active.txt"})
                    ],
                ),
                active_tool_message,
            ],
            context_start_index=3,
        )

        result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "active.txt"},
                        "replace_content": "compressed active result",
                    }
                ]
            },
            ToolExecutionContext(message_state=message_state),
        )

        self.assertEqual(result, "Success")
        self.assertEqual(old_tool_message.compressions, [])
        self.assertEqual(active_tool_message.compressions, ["compressed active result"])

    def test_compress_tool_requires_context_and_valid_replacements(self):
        result_without_context = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "known.txt"},
                        "replace_content": "short",
                    }
                ]
            }
        )
        result_without_replacements = CompressTool().execute(
            {},
            ToolExecutionContext(message_state=MessageState(history=[SystemMessage(content="sys")])),
        )

        self.assertEqual(
            result_without_context,
            "Error: compress_tool requires tool execution context",
        )
        self.assertIn('"replacements" must be a non-empty list', result_without_replacements)

    def test_compress_tool_rejects_unknown_and_duplicate_tool_arguments(self):
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "known.txt"})
                    ],
                ),
                ToolMessage(tool_call_id="call-1", content="old result"),
            ]
        )
        context = ToolExecutionContext(message_state=message_state)

        unknown_result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "missing.txt"},
                        "replace_content": "short",
                    }
                ]
            },
            context,
        )
        duplicate_result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "known.txt"},
                        "replace_content": "short 1",
                    },
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "known.txt"},
                        "replace_content": "short 2",
                    },
                ]
            },
            context,
        )

        self.assertIn("no previous tool result matches read_file", unknown_result)
        self.assertIn("duplicate tool_name/tool_arguments in replacements", duplicate_result)
        self.assertEqual(message_state.history[2].compressions, [])

    def test_compress_tool_compresses_earlier_duplicate_argument_matches(self):
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "same.txt"}),
                        ToolCall(id="call-2", name="read_file", arguments={"path": "same.txt"}),
                    ],
                ),
                ToolMessage(tool_call_id="call-1", content="first result"),
                ToolMessage(tool_call_id="call-2", content="second result"),
            ]
        )

        result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "same.txt"},
                        "replace_content": "short",
                    }
                ]
            },
            ToolExecutionContext(message_state=message_state),
        )

        self.assertEqual(result, "Success")
        self.assertEqual(message_state.history[2].compressions, ["short"])
        self.assertEqual(message_state.history[3].compressions, [])

    def test_compress_tool_keeps_last_uncompressed_duplicate_match(self):
        first_tool_message = ToolMessage(
            tool_call_id="call-1",
            content="first result",
            compressions=["already short"],
        )
        second_tool_message = ToolMessage(tool_call_id="call-2", content="second result")
        third_tool_message = ToolMessage(tool_call_id="call-3", content="third result")
        message_state = MessageState(
            history=[
                SystemMessage(content="sys"),
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "same.txt"}),
                        ToolCall(id="call-2", name="read_file", arguments={"path": "same.txt"}),
                        ToolCall(id="call-3", name="read_file", arguments={"path": "same.txt"}),
                    ],
                ),
                first_tool_message,
                second_tool_message,
                third_tool_message,
            ]
        )

        result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_name": "read_file",
                        "tool_arguments": {"path": "same.txt"},
                        "replace_content": "short",
                    }
                ]
            },
            ToolExecutionContext(message_state=message_state),
        )

        self.assertEqual(result, "Success")
        self.assertEqual(first_tool_message.compressions, ["already short"])
        self.assertEqual(second_tool_message.compressions, ["short"])
        self.assertEqual(third_tool_message.compressions, [])
