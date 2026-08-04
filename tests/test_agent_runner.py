from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_job.agent import AgentRunner
from ai_job.communication import AssistantMessage, SystemMessage, ToolMessage, UserMessage
from ai_job.infra.logging import LogWrapper
from ai_job.provider_adapters import BaseChatModel
from ai_job.tools import BaseTool, ToolCall, ToolExecutor, ToolRegistry


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the given text."
    parameters_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def _run(self, arguments):
        return arguments["text"]


class ScriptedChatModel(BaseChatModel):
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen_histories = []

    def complete(self, history, tool_registry):
        self.seen_histories.append(list(history))
        if not self._replies:
            raise AssertionError("ScriptedChatModel has no more replies")
        return self._replies.pop(0)


class AgentRunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        LogWrapper.configure(Path(self._tmp_dir.name) / "trace.log", "none")

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_run_turn_executes_tool_call_then_returns_final_text(self):
        registry = ToolRegistry([EchoTool()])
        model = ScriptedChatModel(
            [
                AssistantMessage(
                    content=None,
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})],
                ),
                AssistantMessage(content="done"),
            ]
        )
        runner = AgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=3,
        )
        history = [SystemMessage(content="sys"), UserMessage(content="please echo")]

        result = runner.run_turn(history)

        self.assertEqual(result, "done")
        self.assertEqual(history[-3].tool_calls[0].name, "echo")
        self.assertIsInstance(history[-2], ToolMessage)
        self.assertEqual(history[-2].tool_call_id, "call-1")
        self.assertEqual(history[-2].content, "hello")
        self.assertEqual(history[-1].content, "done")
        self.assertIsInstance(model.seen_histories[1][-1], ToolMessage)

    def test_run_turn_rejects_final_assistant_without_text(self):
        registry = ToolRegistry([])
        runner = AgentRunner(
            chat_model=ScriptedChatModel([AssistantMessage(content=None)]),
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )

        with self.assertRaisesRegex(RuntimeError, "最终响应缺少文本"):
            runner.run_turn([UserMessage(content="hi")])

    def test_run_turn_stops_after_max_tool_rounds(self):
        registry = ToolRegistry([EchoTool()])
        runner = AgentRunner(
            chat_model=ScriptedChatModel(
                [
                    AssistantMessage(
                        content=None,
                        tool_calls=[
                            ToolCall(id="call-1", name="echo", arguments={"text": "still working"})
                        ],
                    )
                ]
            ),
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )

        with self.assertRaisesRegex(RuntimeError, "工具调用轮数超过上限"):
            runner.run_turn([UserMessage(content="hi")])
