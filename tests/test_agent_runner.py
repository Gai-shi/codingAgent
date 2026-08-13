from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_job.agent import AgentRunner
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


class PathEchoTool(BaseTool):
    name = "read_file"
    description = "Return the given path."
    parameters_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def _run(self, arguments):
        return arguments["path"]


class ScriptedChatModel(BaseChatModel):
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen_histories = []

    def complete(self, history, tool_registry):
        self.seen_histories.append(list(history))
        if not self._replies:
            raise AssertionError("ScriptedChatModel has no more replies")
        return self._replies.pop(0)


class RecordingCompressionManager:
    def __init__(self):
        self.seen_histories = []

    def compress_if_needed(self, history):
        self.seen_histories.append(list(history))


class AgentRunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        LogWrapper.configure(Path(self._tmp_dir.name) / "trace.log", "none")
        SessionRecorder.configure(Path(self._tmp_dir.name) / "sessions" / "sessions.md")

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

        result = runner.run_turn(MessageState(history=history))

        self.assertEqual(result, "done")
        self.assertEqual(history[-3].tool_calls[0].name, "echo")
        self.assertIsInstance(history[-2], ToolMessage)
        self.assertEqual(history[-2].tool_call_id, "call-1")
        self.assertEqual(history[-2].content, "hello")
        self.assertEqual(history[-1].content, "done")
        self.assertIsInstance(model.seen_histories[1][-1], ToolMessage)

        session_text = SessionRecorder.session_path().read_text(encoding="utf-8")
        self.assertIn("## ", session_text)
        self.assertIn("AssistantMessage", session_text)
        self.assertIn("ToolCall echo", session_text)
        self.assertIn('"text": "hello"', session_text)
        self.assertIn("ToolResult echo", session_text)
        self.assertIn("hello", session_text)

    def test_run_turn_logs_tool_call_index_count_and_path(self):
        registry = ToolRegistry([PathEchoTool()])
        model = ScriptedChatModel(
            [
                AssistantMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                        ToolCall(id="call-2", name="read_file", arguments={"path": "b.py"}),
                    ],
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

        history = [SystemMessage(content="sys"), UserMessage(content="please read files")]

        runner.run_turn(MessageState(history=history))

        log_text = LogWrapper.log_path().read_text(encoding="utf-8")
        self.assertIn(
            "DEBUG [trace] round=1 tool_call=1/2 tool=read_file path=a.py",
            log_text,
        )
        self.assertIn(
            "DEBUG [trace] round=1 tool_call=2/2 tool=read_file path=b.py",
            log_text,
        )

    def test_run_turn_sends_system_and_messages_from_context_start_index(self):
        registry = ToolRegistry([])
        model = ScriptedChatModel([AssistantMessage(content="done")])
        runner = AgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )
        old_user_message = UserMessage(content="old user")
        history = [
            SystemMessage(content="sys"),
            old_user_message,
            AssistantMessage(content="old assistant"),
            UserMessage(content="active user"),
        ]

        runner.run_turn(MessageState(history=history, context_start_index=3))

        self.assertEqual(model.seen_histories[0][0], SystemMessage(content="sys"))
        self.assertNotIn(old_user_message, model.seen_histories[0])
        self.assertEqual(model.seen_histories[0][1], UserMessage(content="active user"))

    def test_run_turn_invokes_compression_before_each_model_request(self):
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
        compression_manager = RecordingCompressionManager()
        runner = AgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=3,
            compression_manager=compression_manager,
        )
        history = [SystemMessage(content="sys"), UserMessage(content="please echo")]

        runner.run_turn(MessageState(history=history))

        self.assertEqual(len(compression_manager.seen_histories), 2)
        self.assertEqual(
            compression_manager.seen_histories[0],
            [SystemMessage(content="sys"), UserMessage(content="please echo")],
        )
        self.assertIsInstance(compression_manager.seen_histories[1][-1], ToolMessage)

    def test_run_turn_uses_compressed_history_for_model_request(self):
        registry = ToolRegistry([])
        model = ScriptedChatModel([AssistantMessage(content="done")])

        class RewritingCompressionManager:
            def compress_if_needed(self, history):
                history[:] = [
                    history[0],
                    SummaryMessage(complete_turn_summary="compressed"),
                    history[-1],
                ]

        runner = AgentRunner(
            chat_model=model,
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
            compression_manager=RewritingCompressionManager(),
        )
        old_user_message = UserMessage(content="old user")
        active_user_message = UserMessage(content="active user")
        history = [
            SystemMessage(content="sys"),
            old_user_message,
            AssistantMessage(content="old answer"),
            active_user_message,
        ]

        runner.run_turn(MessageState(history=history))

        self.assertEqual(
            model.seen_histories[0],
            [
                SystemMessage(content="sys"),
                SummaryMessage(complete_turn_summary="compressed"),
                active_user_message,
            ],
        )
        self.assertNotIn(old_user_message, model.seen_histories[0])

    def test_run_turn_rejects_final_assistant_without_text(self):
        registry = ToolRegistry([])
        runner = AgentRunner(
            chat_model=ScriptedChatModel([AssistantMessage(content=None)]),
            tool_registry=registry,
            tool_executor=ToolExecutor(registry),
            max_tool_rounds=1,
        )

        with self.assertRaisesRegex(RuntimeError, "最终响应缺少文本"):
            runner.run_turn(MessageState(history=[UserMessage(content="hi")]))

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
            runner.run_turn(MessageState(history=[UserMessage(content="hi")]))
