from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import sentinel, patch

from ai_job.composition import build_initial_messages, create_cli_runtime
from ai_job.infra.env import AppEnv


def make_app_env() -> AppEnv:
    return AppEnv(
        openai_api_key="key",
        openai_model="model",
        openai_base_url="http://localhost:8787/v1",
        timeout_seconds=60.0,
        max_tool_rounds=8,
        context_window_override=None,
        compaction_reserve_tokens=16384,
        compaction_keep_recent_tokens=20000,
        system_prompt="system prompt",
        filter_terminal_log_level="none",
    )


class CliFactoryTest(unittest.TestCase):
    def test_build_initial_messages_includes_workspace_root(self):
        workspace = Path("/tmp/example-workspace")

        messages = build_initial_messages(make_app_env(), workspace)

        self.assertEqual(len(messages), 1)
        self.assertIn("system prompt", messages[0].content)
        self.assertIn("Current workspace root: /tmp/example-workspace", messages[0].content)

    def test_create_cli_runtime_assembles_dependencies(self):
        app_env = make_app_env()
        workspace = Path("/tmp/example-workspace")

        with patch("ai_job.composition.cli_factory.create_default_tool_registry") as registry_mock:
            with patch("ai_job.composition.cli_factory.ToolExecutor") as executor_mock:
                with patch("ai_job.composition.cli_factory.OpenAIModel") as model_mock:
                    with patch(
                        "ai_job.composition.cli_factory.create_compression_manager",
                        return_value=sentinel.compression_manager,
                    ) as compression_manager_mock:
                        with patch("ai_job.composition.cli_factory.AgentRunner") as agent_runner_mock:
                            runtime = create_cli_runtime(
                                app_env=app_env,
                                workspace_root=workspace,
                                request_protected_grep_approval=sentinel.approval_callback,
                                include_compress_tool=False,
                            )

        registry_mock.assert_called_once_with(
            workspace,
            sentinel.approval_callback,
            include_compress_tool=False,
        )
        executor_mock.assert_called_once_with(registry_mock.return_value)
        model_mock.assert_called_once_with(app_env)
        compression_manager_mock.assert_called_once_with(app_env, model_mock.return_value)
        agent_runner_mock.assert_called_once_with(
            chat_model=model_mock.return_value,
            tool_registry=registry_mock.return_value,
            tool_executor=executor_mock.return_value,
            max_tool_rounds=app_env.max_tool_rounds,
            compression_manager=sentinel.compression_manager,
        )
        expected_content = build_initial_messages(app_env, workspace)[0].content
        self.assertEqual(runtime.message_state.history[0].content, expected_content)
        self.assertIs(runtime.tool_registry, registry_mock.return_value)
        self.assertIs(runtime.agent_runner, agent_runner_mock.return_value)


if __name__ == "__main__":
    unittest.main()
