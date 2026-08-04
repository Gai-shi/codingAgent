from __future__ import annotations

import json
import unittest

from ai_job.communication import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    message_history_to_debug_dicts,
)
from ai_job.infra.env import AppEnv
from ai_job.provider_adapters import OpenAIModel
from ai_job.tool_adapters import OpenAIToolCallAdapter
from ai_job.tools import BaseTool, ToolCall, ToolRegistry


class ExampleTool(BaseTool):
    name = "example"
    description = "Example tool."
    parameters_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def _run(self, arguments):
        return arguments["text"]


def make_app_env():
    return AppEnv(
        openai_api_key="key",
        openai_model="model",
        openai_base_url="http://example.test/v1",
        timeout_seconds=1.0,
        max_tool_rounds=2,
        system_prompt="system",
        filter_terminal_log_level="none",
    )


class MessageDebugTest(unittest.TestCase):
    def test_message_history_to_debug_dicts_preserves_roles_and_tool_calls(self):
        history = [
            SystemMessage(content="system"),
            UserMessage(content="user"),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(id="call-1", name="example", arguments={"text": "你好"})],
            ),
            ToolMessage(tool_call_id="call-1", content="result"),
        ]

        result = message_history_to_debug_dicts(history)

        self.assertEqual(result[0], {"role": "system", "content": "system"})
        self.assertEqual(result[1], {"role": "user", "content": "user"})
        self.assertEqual(result[2]["role"], "assistant")
        self.assertEqual(result[2]["tool_calls"][0]["arguments"], {"text": "你好"})
        self.assertEqual(
            result[3],
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        )


class OpenAIToolCallAdapterTest(unittest.TestCase):
    def test_render_tool_definitions_deep_copies_parameter_schema(self):
        adapter = OpenAIToolCallAdapter()
        tool = ExampleTool()
        registry = ToolRegistry([tool])

        rendered = adapter.render_tool_definitions(registry)
        rendered[0]["function"]["parameters"]["properties"]["text"]["type"] = "integer"

        self.assertEqual(tool.parameters_schema["properties"]["text"]["type"], "string")

    def test_parse_and_render_tool_call_round_trip_arguments(self):
        adapter = OpenAIToolCallAdapter()
        raw_tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {"name": "example", "arguments": '{"text": "你好"}'},
        }

        parsed = adapter.parse_tool_call(raw_tool_call)
        rendered = adapter.render_tool_call(parsed)

        self.assertEqual(parsed, ToolCall(id="call-1", name="example", arguments={"text": "你好"}))
        self.assertEqual(rendered["id"], "call-1")
        self.assertEqual(rendered["function"]["name"], "example")
        self.assertEqual(json.loads(rendered["function"]["arguments"]), {"text": "你好"})

    def test_parse_tool_call_rejects_non_object_arguments(self):
        adapter = OpenAIToolCallAdapter()

        with self.assertRaisesRegex(ValueError, "expected a JSON object"):
            adapter.parse_tool_call(
                {
                    "id": "call-1",
                    "function": {"name": "example", "arguments": '["not", "object"]'},
                }
            )


class OpenAIModelTest(unittest.TestCase):
    def test_parse_assistant_message_accepts_text_response(self):
        model = OpenAIModel(make_app_env())
        response_body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

        result = model._parse_assistant_message(response_body)

        self.assertEqual(result, AssistantMessage(content="hi", tool_calls=[]))

    def test_parse_assistant_message_accepts_tool_calls_without_text(self):
        model = OpenAIModel(make_app_env())
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "example",
                                        "arguments": '{"text": "hi"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

        result = model._parse_assistant_message(response_body)

        self.assertEqual(result.content, None)
        self.assertEqual(result.tool_calls, [ToolCall(id="call-1", name="example", arguments={"text": "hi"})])

    def test_parse_assistant_message_rejects_missing_choices(self):
        model = OpenAIModel(make_app_env())

        with self.assertRaisesRegex(RuntimeError, "缺少 choices"):
            model._parse_assistant_message("{}")

    def test_render_message_uses_openai_chat_completions_roles(self):
        model = OpenAIModel(make_app_env())

        assistant = AssistantMessage(
            content=None,
            tool_calls=[ToolCall(id="call-1", name="example", arguments={"text": "hi"})],
        )

        self.assertEqual(model._render_message(SystemMessage("sys")), {"role": "system", "content": "sys"})
        self.assertEqual(model._render_message(UserMessage("hi")), {"role": "user", "content": "hi"})
        self.assertEqual(
            model._render_message(ToolMessage(tool_call_id="call-1", content="ok")),
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        )
        rendered_assistant = model._render_message(assistant)
        self.assertEqual(rendered_assistant["role"], "assistant")
        self.assertEqual(rendered_assistant["content"], None)
        self.assertEqual(rendered_assistant["tool_calls"][0]["id"], "call-1")
