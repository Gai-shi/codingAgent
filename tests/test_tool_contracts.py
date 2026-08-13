from __future__ import annotations

import unittest

from ai_job.communication import MessageState, SystemMessage, ToolMessage, UserMessage
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
                old_tool_message,
                UserMessage(content="active"),
                active_tool_message,
            ],
            context_start_index=2,
        )

        result = CompressTool().execute(
            {
                "replacements": [
                    {
                        "tool_call_id": "call-1",
                        "replace_content": "compressed active result",
                    }
                ]
            },
            ToolExecutionContext(message_state=message_state),
        )

        self.assertEqual(result, "Success")
        self.assertEqual(old_tool_message.compressions, [])
        self.assertEqual(active_tool_message.compressions, ["compressed active result"])
