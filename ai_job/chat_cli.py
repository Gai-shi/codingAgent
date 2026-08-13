"""A minimal terminal coding-agent CLI.

当前文件只负责：
- 从项目级 .env 和环境变量读取配置；
- 通过 composition 创建 CLI 运行时对象；
- 编排当前进程内存里的消息历史；
- 处理终端输入输出。

agent loop、provider 请求解析、tool calling 协议转换均已拆到独立模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from .composition import CliSessionLifecycle, create_cli_runtime
from .communication import (
    MessageHistory,
    UserMessage,
    message_history_to_debug_dicts,
)
from .infra.env import AppEnv, EnvLoader
from .infra.logging import LogWrapper
from .infra.session_recording import SessionRecorder
from .terminal_input import AllowInputEcho, SuppressInputEchoAndDiscard
from .tools import ToolRegistry


EXIT_COMMANDS = {"exit", "quit", "et", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE_PATH = APP_ROOT / ".env"
DEFAULT_TRACE_LOG_PATH = APP_ROOT / ".ai_job" / "logs" / "log.log"
DEFAULT_SESSION_RECORD_PATH = APP_ROOT / ".ai_job" / "sessions" / "sessions.md"


def enable_line_editing() -> None:
    """Enable readline-backed input editing when the platform provides it."""
    try:
        import readline  # noqa: F401
    except ImportError:
        return


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ai-job 最小 coding agent CLI")
    parser.add_argument(
        "-w",
        "--workspace",
        default=None,
        help="agent 读写代码的工作区目录。默认使用启动命令时的当前目录。",
    )
    parser.add_argument(
        "--disable-compress-tool",
        action="store_true",
        help="不注册 compress_tool，用于对比评测工具输出压缩的效果。",
    )
    return parser.parse_args(argv)


def resolve_workspace_root(path_text: Optional[str]) -> Path:
    raw_path = Path(path_text).expanduser() if path_text else Path.cwd()
    resolved = raw_path.resolve()
    if not resolved.exists():
        raise ValueError(f"workspace 不存在：{resolved}")
    if not resolved.is_dir():
        raise ValueError(f"workspace 不是目录：{resolved}")
    return resolved


def default_trace_log_path() -> Path:
    return DEFAULT_TRACE_LOG_PATH


def default_session_record_path() -> Path:
    return DEFAULT_SESSION_RECORD_PATH


def create_protected_grep_approval(workspace_root: Path) -> Callable[[Path], bool]:
    def request_protected_grep_approval(path: Path) -> bool:
        relative_path = path.relative_to(workspace_root) if path != workspace_root else Path(".")
        print()
        print("grep 请求检索隐藏目录或保护目录。")
        print(f"范围：{relative_path}")
        with AllowInputEcho():
            answer = input("如果你同意本次检索，请输入 yes；其它输入表示拒绝> ").strip().lower()
        return answer == "yes"

    return request_protected_grep_approval


def print_banner(app_env: AppEnv, tool_registry: ToolRegistry, workspace_root: Path) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {app_env.openai_model}")
    print(f"base_url: {app_env.openai_base_url}")
    print(f"workspace: {workspace_root}")
    print(f"log_file: {LogWrapper.log_path()}")
    print(f"session_record: {SessionRecorder.session_path()}")
    print(f"terminal_log_level: {LogWrapper.filter_terminal_log_level()}")
    print(f"tools: {', '.join(tool_registry.names())}")
    print("输入 /context 查看当前模型上下文。")
    print("输入 exit / quit / et / Ctrl-D 退出。")
    print()


def print_context(messages: MessageHistory) -> None:
    print()
    print(json.dumps(message_history_to_debug_dicts(messages), ensure_ascii=False, indent=2))
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_cli_args(argv)
        workspace_root = resolve_workspace_root(args.workspace)
        app_env = EnvLoader.load(DEFAULT_ENV_FILE_PATH)
        session_lifecycle = CliSessionLifecycle(
            trace_log_path=default_trace_log_path(),
            session_record_path=default_session_record_path(),
        )
        session_lifecycle.start(app_env=app_env, workspace_root=workspace_root)
    except ValueError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "示例：OPENAI_API_KEY=xxx OPENAI_MODEL=xxx python3 -m ai_job --workspace /path/to/project",
            file=sys.stderr,
        )
        return 2

    runtime = create_cli_runtime(
        app_env=app_env,
        workspace_root=workspace_root,
        request_protected_grep_approval=create_protected_grep_approval(workspace_root),
        include_compress_tool=not args.disable_compress_tool,
    )
    message_state = runtime.message_state
    agent_runner = runtime.agent_runner
    session_lifecycle.record_initial_system_message(message_state)
    print_banner(app_env, runtime.tool_registry, workspace_root)
    enable_line_editing()

    while True:
        try:
            user_text = input("你> ").strip()
        except EOFError:
            session_lifecycle.record_exit_snapshot(message_state)
            print("\n再见。")
            return 0
        except KeyboardInterrupt:
            print("\n已中断。")
            return 130

        if not user_text:
            continue

        if user_text.lower() in EXIT_COMMANDS:
            session_lifecycle.record_exit_snapshot(message_state)
            print("再见。")
            return 0

        if user_text.lower() in CONTEXT_COMMANDS:
            print_context(message_state.model_visible_history())
            continue

        turn_start = len(message_state.history)
        turn_context_start_index = message_state.context_start_index
        message_state.history.append(UserMessage(content=user_text))
        SessionRecorder.record_session("UserMessage", user_text, "text")
        try:
            with SuppressInputEchoAndDiscard():
                assistant_text = agent_runner.run_turn(message_state)
        except KeyboardInterrupt:
            del message_state.history[turn_start:]
            message_state.context_start_index = turn_context_start_index
            print("\n已中断。")
            return 130
        except RuntimeError as exc:
            del message_state.history[turn_start:]
            message_state.context_start_index = turn_context_start_index
            print(f"错误：{exc}", file=sys.stderr)
            continue

        print(f"\n助手> {assistant_text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
