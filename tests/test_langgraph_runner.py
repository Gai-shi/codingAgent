from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from langchain_core.messages import AIMessage as LangChainAIMessage
    from langchain_core.messages import HumanMessage as LangChainHumanMessage
    from langchain_core.messages import SystemMessage as LangChainSystemMessage
    from langchain_core.messages import ToolMessage as LangChainToolMessage
except ImportError:  # pragma: no cover - exercised only without optional deps.
    LangChainAIMessage = None
    LangChainHumanMessage = None
    LangChainSystemMessage = None
    LangChainToolMessage = None

from ai_job.communication import (
    AssistantMessage,
    MessageState,
    SummaryMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ai_job.infra.logging import LogWrapper
from ai_job.infra.session_recording import SessionRecorder
from ai_job.tools import BaseTool, CompressTool, ToolCall, ToolExecutor, ToolRegistry


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


class FakeLangChainModel:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen_messages = []
        self.bound_tool_names = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.seen_messages.append(list(messages))
        if not self._replies:
            raise AssertionError("FakeLangChainModel has no more replies")
        return self._replies.pop(0)


@unittest.skipIf(LangChainAIMessage is None, "langgraph optional dependencies are not installed")
class LangGraphRunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        LogWrapper.configure(Path(self._tmp_dir.name) / "trace.log", "none")
        SessionRecorder.configure(Path(self._tmp_dir.name) / "sessions" / "sessions.md")

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_internal_messages_convert_to_langchain_messages(self):
        from ai_job.agent.langgraph_runner.message_conversion import (
            internal_message_to_langchain_message,
        )

        messages = [
            SystemMessage(content="sys"),
            UserMessage(content="hello"),
            SummaryMessage(complete_turn_summary="old turns", split_turn_summary="partial turn"),
            AssistantMessage(
                content=None,
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
            ),
            ToolMessage(tool_call_id="call-1", content="tool result", compressions=["short result"]),
        ]

        converted = [internal_message_to_langchain_message(message) for message in messages]

        self.assertIsInstance(converted[0], LangChainSystemMessage)
        self.assertEqual(converted[0].content, "sys")
        self.assertIsInstance(converted[1], LangChainHumanMessage)
        self.assertEqual(converted[1].content, "hello")
        self.assertIsInstance(converted[2], LangChainHumanMessage)
        self.assertIn("完整回合摘要：\nold turns", converted[2].content)
        self.assertIn("当前未完整保留回合的前半段摘要：\npartial turn", converted[2].content)
        self.assertIsInstance(converted[3], LangChainAIMessage)
        self.assertEqual(converted[3].content, "")
        self.assertEqual(
            converted[3].tool_calls,
            [{"name": "echo", "args": {"text": "hi"}, "id": "call-1", "type": "tool_call"}],
        )
        self.assertIsInstance(converted[4], LangChainToolMessage)
        self.assertEqual(converted[4].tool_call_id, "call-1")
        self.assertEqual(converted[4].content, "short result")

    def test_langchain_ai_message_converts_to_internal_assistant_message(self):
        from ai_job.agent.langgraph_runner.message_conversion import (
            langchain_ai_message_to_internal_assistant_message,
        )

        converted = langchain_ai_message_to_internal_assistant_message(
            LangChainAIMessage(
                content="working",
                tool_calls=[
                    {"id": "call-1", "name": "echo", "args": {"text": "hi"}},
                    {"id": "call-2", "name": "read_file", "args": {"path": "a.py"}},
                ],
            )
        )

        self.assertEqual(
            converted,
            AssistantMessage(
                content="working",
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"text": "hi"}),
                    ToolCall(id="call-2", name="read_file", arguments={"path": "a.py"}),
                ],
            ),
        )

    def test_wrap_tool_calls_internal_base_tool(self):
        from ai_job.agent.langgraph_runner.tool_bridge import wrap_tool

        wrapped = wrap_tool(EchoTool())

        self.assertEqual(wrapped.name, "echo")
        self.assertEqual(wrapped.description, "Return the given text.")
        self.assertEqual(wrapped.invoke({"text": "hello"}), "hello")

    def test_run_turn_returns_final_text_without_tool_calls(self):
        from ai_job.agent.langgraph_runner import LangGraphAgentRunner

        registry = ToolRegistry([])
        model = FakeLangChainModel([LangChainAIMessage(content="done")])
        runner = LangGraphAgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )
        history = [SystemMessage(content="sys"), UserMessage(content="hi")]

        result = runner.run_turn(MessageState(history=history))

        self.assertEqual(result, "done")
        self.assertEqual(history[-1], AssistantMessage(content="done"))

    def test_run_turn_executes_tool_call_then_returns_final_text(self):
        from ai_job.agent.langgraph_runner import LangGraphAgentRunner

        registry = ToolRegistry([EchoTool()])
        model = FakeLangChainModel(
            [
                LangChainAIMessage(
                    content="",
                    tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "hello"}}],
                ),
                LangChainAIMessage(content="done"),
            ]
        )
        runner = LangGraphAgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=3,
        )
        history = [SystemMessage(content="sys"), UserMessage(content="please echo")]

        result = runner.run_turn(MessageState(history=history))

        self.assertEqual(result, "done")
        self.assertEqual(model.bound_tool_names, ["echo"])
        self.assertIsInstance(history[-2], ToolMessage)
        self.assertEqual(history[-2].tool_call_id, "call-1")
        self.assertEqual(history[-2].content, "hello")
        self.assertEqual(history[-1], AssistantMessage(content="done"))

    def test_run_turn_stops_after_max_tool_rounds(self):
        from ai_job.agent.langgraph_runner import LangGraphAgentRunner

        registry = ToolRegistry([EchoTool()])
        model = FakeLangChainModel(
            [
                LangChainAIMessage(
                    content="",
                    tool_calls=[{"id": "call-1", "name": "echo", "args": {"text": "again"}}],
                )
            ]
        )
        runner = LangGraphAgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )

        with self.assertRaisesRegex(RuntimeError, "工具调用轮数超过上限：1"):
            runner.run_turn(
                MessageState(history=[SystemMessage(content="sys"), UserMessage(content="hi")])
            )

    def test_run_turn_passes_message_state_to_compress_tool(self):
        from ai_job.agent.langgraph_runner import LangGraphAgentRunner

        registry = ToolRegistry([CompressTool()])
        model = FakeLangChainModel(
            [
                LangChainAIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "compress-1",
                            "name": "compress_tool",
                            "args": {
                                "replacements": [
                                    {
                                        "tool_name": "read_file",
                                        "tool_arguments": {"path": "large.txt"},
                                        "replace_content": "short",
                                    }
                                ]
                            },
                        }
                    ],
                ),
                LangChainAIMessage(content="done"),
            ]
        )
        runner = LangGraphAgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=2,
        )
        large_tool_message = ToolMessage(tool_call_id="call-old", content="large content")
        history = [
            SystemMessage(content="sys"),
            UserMessage(content="please compress"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(id="call-old", name="read_file", arguments={"path": "large.txt"})
                ],
            ),
            large_tool_message,
        ]

        result = runner.run_turn(MessageState(history=history))

        self.assertEqual(result, "done")
        self.assertEqual(large_tool_message.compressions, ["short"])

    def test_checkpoint_demo_graph_runs_with_thread_id(self):
        from ai_job.agent.langgraph_runner.checkpoint_demo import build_checkpoint_demo_graph

        graph = build_checkpoint_demo_graph()
        result = graph.invoke({"counter": 1}, config={"configurable": {"thread_id": "demo"}})

        self.assertEqual(result, {"counter": 2})


if __name__ == "__main__":
    unittest.main()
