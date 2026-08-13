"""Runtime composition for the terminal CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..agent import AgentRunner
from ..communication import MessageHistory, MessageState, SystemMessage
from ..infra.env import AppEnv
from ..provider_adapters import OpenAIModel
from ..tools import ToolExecutor, ToolRegistry, create_default_tool_registry
from .compression_factory import create_compression_manager


@dataclass(frozen=True)
class CliRuntime:
    message_state: MessageState
    tool_registry: ToolRegistry
    agent_runner: AgentRunner


def create_cli_runtime(
    *,
    app_env: AppEnv,
    workspace_root: Path,
    request_protected_grep_approval: Callable[[Path], bool],
    include_compress_tool: bool = True,
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
    agent_runner = AgentRunner(
        chat_model=chat_model,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        max_tool_rounds=app_env.max_tool_rounds,
        compression_manager=compression_manager,
    )
    return CliRuntime(
        message_state=message_state,
        tool_registry=tool_registry,
        agent_runner=agent_runner,
    )


def build_initial_messages(app_env: AppEnv, workspace_root: Path) -> MessageHistory:
    system_prompt = f"{app_env.system_prompt}\n\nCurrent workspace root: {workspace_root}"
    return [SystemMessage(content=system_prompt)]
