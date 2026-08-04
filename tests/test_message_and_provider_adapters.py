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
from ai_job.infra.http import BaseHttpClient, HttpClientError
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


class FakeHttpClient(BaseHttpClient):
    def __init__(self, response_body, error=None):
        self._response_body = response_body
        self._error = error
        self.calls = []

    def post_json(self, url, payload, headers, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self._error is not None:
            raise self._error
        return self._response_body


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
    def test_complete_posts_rendered_payload_and_parses_text_response(self):
        fake_http_client = FakeHttpClient(
            json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi"}}]})
        )
        model = OpenAIModel(make_app_env(), http_client=fake_http_client)
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(id="call-1", name="example", arguments={"text": "question"})],
            ),
            ToolMessage(tool_call_id="call-1", content="tool result"),
        ]
        registry = ToolRegistry([ExampleTool()])

        result = model.complete(history, registry)

        self.assertEqual(result, AssistantMessage(content="hi", tool_calls=[]))
        self.assertEqual(len(fake_http_client.calls), 1)
        call = fake_http_client.calls[0]
        self.assertEqual(call["url"], "http://example.test/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer key")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["timeout_seconds"], 1.0)
        self.assertEqual(call["payload"]["model"], "model")
        self.assertEqual(call["payload"]["tool_choice"], "auto")
        self.assertEqual(
            call["payload"]["messages"][0],
            {"role": "system", "content": "sys"},
        )
        self.assertEqual(
            call["payload"]["messages"][1],
            {"role": "user", "content": "hi"},
        )
        self.assertEqual(call["payload"]["messages"][2]["role"], "assistant")
        self.assertEqual(call["payload"]["messages"][2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(
            call["payload"]["messages"][3],
            {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
        )
        self.assertEqual(call["payload"]["tools"][0]["function"]["name"], "example")

    def test_complete_parses_tool_calls_without_text(self):
        fake_http_client = FakeHttpClient(
            json.dumps(
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
        )
        model = OpenAIModel(make_app_env(), http_client=fake_http_client)

        result = model.complete([UserMessage(content="hi")], ToolRegistry([]))

        self.assertEqual(result.content, None)
        self.assertEqual(
            result.tool_calls,
            [ToolCall(id="call-1", name="example", arguments={"text": "hi"})],
        )

    def test_complete_rejects_missing_choices(self):
        model = OpenAIModel(make_app_env(), http_client=FakeHttpClient("{}"))

        with self.assertRaisesRegex(RuntimeError, "缺少 choices"):
            model.complete([UserMessage(content="hi")], ToolRegistry([]))

    def test_complete_wraps_http_client_errors_as_llm_request_failures(self):
        model = OpenAIModel(
            make_app_env(),
            http_client=FakeHttpClient(response_body="", error=HttpClientError("network down")),
        )

        with self.assertRaisesRegex(RuntimeError, "LLM 请求失败：network down"):
            model.complete([UserMessage(content="hi")], ToolRegistry([]))
