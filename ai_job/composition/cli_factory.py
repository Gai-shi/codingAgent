"""Runtime composition for the terminal CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..agent import AgentRunner, BaseAgentRunner
from ..communication import MessageHistory, MessageState, SystemMessage
from ..infra.env import AppEnv
from ..provider_adapters import OpenAIModel
from ..tools import ToolExecutor, ToolRegistry, create_default_tool_registry
from .compression_factory import create_compression_manager


RunnerKind = Literal["native", "langgraph"]
LANGGRAPH_INSTALL_MESSAGE = "请先安装 LangGraph 可选依赖：python3 -m pip install -e '.[langgraph]'"


@dataclass(frozen=True)
class CliRuntime:
    message_state: MessageState
    tool_registry: ToolRegistry
    agent_runner: BaseAgentRunner


def create_cli_runtime(
    *,
    app_env: AppEnv,
    workspace_root: Path,
    request_protected_grep_approval: Callable[[Path], bool],
    include_compress_tool: bool = True,
    runner_kind: RunnerKind = "native",
) -> CliRuntime:
    message_state = MessageState(history=build_initial_messages(app_env, workspace_root))
    tool_registry = create_default_tool_registry(
        workspace_root,
        request_protected_grep_approval,
        include_compress_tool=include_compress_tool,
    )
    tool_executor = ToolExecutor(tool_registry)
    chat_model = OpenAIModel(app_env)
    compression_manager = create_compression_manager(app_env, chat_model)
    agent_runner = create_agent_runner(
        runner_kind=runner_kind,
        app_env=app_env,
        native_chat_model=chat_model,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        compression_manager=compression_manager,
    )
    return CliRuntime(
        message_state=message_state,
        tool_registry=tool_registry,
        agent_runner=agent_runner,
    )


def create_agent_runner(
    *,
    runner_kind: RunnerKind,
    app_env: AppEnv,
    native_chat_model: OpenAIModel,
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    compression_manager,
) -> BaseAgentRunner:
    if runner_kind == "native":
        return AgentRunner(
            chat_model=native_chat_model,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            max_tool_rounds=app_env.max_tool_rounds,
            compression_manager=compression_manager,
        )

    if runner_kind == "langgraph":
        try:
            from langchain_openai import ChatOpenAI

            from ..agent.langgraph_runner import LangGraphAgentRunner
        except ModuleNotFoundError as exc:
            if exc.name in {"langchain_openai", "langchain_core", "langgraph"}:
                raise ValueError(LANGGRAPH_INSTALL_MESSAGE) from exc
            raise

        return LangGraphAgentRunner(
            chat_model=ChatOpenAI(
                model=app_env.openai_model,
                api_key=app_env.openai_api_key,
                base_url=app_env.openai_base_url,
                timeout=app_env.timeout_seconds,
            ),
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            max_tool_rounds=app_env.max_tool_rounds,
            compression_manager=compression_manager,
        )

    raise ValueError(f"未知 runner：{runner_kind}")


def build_initial_messages(app_env: AppEnv, workspace_root: Path) -> MessageHistory:
    system_prompt = f"{app_env.system_prompt}\n\nCurrent workspace root: {workspace_root}"
    return [SystemMessage(content=system_prompt)]
