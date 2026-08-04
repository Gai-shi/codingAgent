"""A minimal terminal coding-agent CLI.

当前文件只负责：
- 从项目级 .env 和环境变量读取配置；
- 组装 CLI 需要的 agent / provider / tool 对象；
- 维护当前进程内存里的消息历史；
- 处理终端输入输出。

agent loop、provider 请求解析、tool calling 协议转换均已拆到独立模块。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .agent import AgentRunner
from .communication import (
    MessageHistory,
    SystemMessage,
    UserMessage,
    message_history_to_debug_dicts,
)
from .infra.env import AppEnv, EnvLoader
from .infra.logging import LogWrapper
from .provider_adapters import OpenAIModel
from .terminal_input import AllowInputEcho, SuppressInputEchoAndDiscard
from .tools import ToolExecutor, ToolRegistry, create_default_tool_registry


EXIT_COMMANDS = {"exit", "quit", "et", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE_PATH = WORKSPACE_ROOT / ".env"
DEFAULT_TRACE_LOG_PATH = WORKSPACE_ROOT / ".ai_job" / "trace.log"


def request_protected_grep_approval(path: Path) -> bool:
    relative_path = path.relative_to(WORKSPACE_ROOT) if path != WORKSPACE_ROOT else Path(".")
    print()
    print("grep 请求检索隐藏目录或保护目录。")
    print(f"范围：{relative_path}")
    with AllowInputEcho():
        answer = input("如果你同意本次检索，请输入 yes；其它输入表示拒绝> ").strip().lower()
    return answer == "yes"


def build_initial_messages(app_env: AppEnv) -> MessageHistory:
    return [SystemMessage(content=app_env.system_prompt)]


def print_banner(app_env: AppEnv, tool_registry: ToolRegistry) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {app_env.openai_model}")
    print(f"base_url: {app_env.openai_base_url}")
    print(f"workspace: {WORKSPACE_ROOT}")
    print(f"trace_log: {LogWrapper.log_path()}")
    print(f"terminal_log_level: {LogWrapper.filter_terminal_log_level()}")
    print(f"tools: {', '.join(tool_registry.names())}")
    print("输入 /context 查看当前内存里的 messages。")
    print("输入 exit / quit / et / Ctrl-D 退出。")
    print()


def print_context(messages: MessageHistory) -> None:
    print()
    print(json.dumps(message_history_to_debug_dicts(messages), ensure_ascii=False, indent=2))
    print()


def main() -> int:
    try:
        app_env = EnvLoader.load(DEFAULT_ENV_FILE_PATH)
        LogWrapper.configure(
            log_path=DEFAULT_TRACE_LOG_PATH,
            filter_terminal_log_level=app_env.filter_terminal_log_level,
        )
    except ValueError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "示例：OPENAI_API_KEY=xxx OPENAI_MODEL=xxx python3 -m ai_job",
            file=sys.stderr,
        )
        return 2

    messages = build_initial_messages(app_env)
    tool_registry = create_default_tool_registry(WORKSPACE_ROOT, request_protected_grep_approval)
    tool_executor = ToolExecutor(tool_registry)
    chat_model = OpenAIModel(app_env)
    agent_runner = AgentRunner(
        chat_model=chat_model,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        max_tool_rounds=app_env.max_tool_rounds,
    )
    print_banner(app_env, tool_registry)

    while True:
        try:
            user_text = input("你> ").strip()
        except EOFError:
            print("\n再见。")
            return 0
        except KeyboardInterrupt:
            print("\n已中断。")
            return 130

        if not user_text:
            continue

        if user_text.lower() in EXIT_COMMANDS:
            print("再见。")
            return 0

        if user_text.lower() in CONTEXT_COMMANDS:
            print_context(messages)
            continue

        turn_start = len(messages)
        messages.append(UserMessage(content=user_text))
        try:
            with SuppressInputEchoAndDiscard():
                assistant_text = agent_runner.run_turn(messages)
        except KeyboardInterrupt:
            del messages[turn_start:]
            print("\n已中断。")
            return 130
        except RuntimeError as exc:
            del messages[turn_start:]
            print(f"错误：{exc}", file=sys.stderr)
            continue

        print(f"\n助手> {assistant_text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
