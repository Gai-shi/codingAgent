"""A minimal terminal coding-agent CLI.

当前文件只负责：
- 从项目级 .env 和环境变量读取配置；
- 组装 CLI 需要的 agent / provider / tool 对象；
- 维护当前进程内存里的消息历史；
- 处理终端输入输出。

agent loop、provider 请求解析、tool calling 协议转换均已拆到独立模块。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from .agent import AgentRunner
from .agent.token_counting import estimate_message_tokens
from .compress import CompressionManager, CompressionPlan
from .communication import (
    AssistantMessage,
    MessageHistory,
    SummaryMessage,
    SystemMessage,
    UserMessage,
    message_history_to_debug_dicts,
)
from .infra.env import AppEnv, EnvLoader
from .infra.logging import LogWrapper
from .infra.session_recording import SessionRecorder
from .provider_adapters import BaseChatModel, OpenAIModel, resolve_context_window
from .terminal_input import AllowInputEcho, SuppressInputEchoAndDiscard
from .tools import ToolExecutor, ToolRegistry, create_default_tool_registry


EXIT_COMMANDS = {"exit", "quit", "et", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE_PATH = APP_ROOT / ".env"
DEFAULT_TRACE_LOG_PATH = APP_ROOT / ".ai_job" / "logs" / "log.log"
DEFAULT_SESSION_RECORD_PATH = APP_ROOT / ".ai_job" / "sessions" / "sessions.md"


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ai-job 最小 coding agent CLI")
    parser.add_argument(
        "-w",
        "--workspace",
        default=None,
        help="agent 读写代码的工作区目录。默认使用启动命令时的当前目录。",
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


def build_initial_messages(app_env: AppEnv, workspace_root: Path) -> MessageHistory:
    system_prompt = f"{app_env.system_prompt}\n\nCurrent workspace root: {workspace_root}"
    return [SystemMessage(content=system_prompt)]


def create_compression_manager(app_env: AppEnv, chat_model: BaseChatModel) -> CompressionManager:
    context_window = resolve_context_window(
        app_env.openai_model,
        context_window_override=app_env.context_window_override,
    )
    return CompressionManager(
        context_window=context_window,
        reserve_tokens=app_env.compaction_reserve_tokens,
        keep_recent_tokens=app_env.compaction_keep_recent_tokens,
        token_counter=estimate_message_tokens,
        summarizer=create_summarizer(chat_model),
    )


def create_summarizer(chat_model: BaseChatModel) -> Callable[[CompressionPlan, MessageHistory], SummaryMessage]:
    empty_tool_registry = ToolRegistry([])

    def summarize(plan: CompressionPlan, history: MessageHistory) -> SummaryMessage:
        assistant_message = chat_model.complete(
            build_summary_messages(plan, history),
            empty_tool_registry,
        )
        return parse_summary_message(assistant_message)

    return summarize


def build_summary_messages(plan: CompressionPlan, history: MessageHistory) -> MessageHistory:
    content = "\n\n".join(
        [
            "请压缩以下对话上下文，输出 JSON，不要输出其它文本。",
            "JSON schema: {\"complete_turn_summary\": string, \"split_turn_summary\": string | null}",
            "complete_messages 是完整压缩掉的旧回合。",
            "split_messages 是被切开回合中未保留的前半段；如果没有该段，请返回 null。",
            "保留事实、用户目标、已做决定、工具调用结果、文件路径、错误信息和仍需继续的事项。",
            "不要编造未出现的信息。",
            "complete_messages:",
            json.dumps(
                message_history_to_debug_dicts(history[plan.complete_range.start : plan.complete_range.end]),
                ensure_ascii=False,
                indent=2,
            ),
            "split_messages:",
            json.dumps(
                message_history_to_debug_dicts(history[plan.split_range.start : plan.split_range.end])
                if plan.split_range is not None
                else [],
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    return [UserMessage(content=content)]


def parse_summary_message(assistant_message: AssistantMessage) -> SummaryMessage:
    if not isinstance(assistant_message.content, str):
        raise RuntimeError("压缩摘要响应缺少文本 content")

    try:
        data = json.loads(assistant_message.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("压缩摘要响应不是合法 JSON") from exc

    if not isinstance(data, dict):
        raise RuntimeError("压缩摘要响应 JSON 必须是对象")

    complete_turn_summary = data.get("complete_turn_summary")
    split_turn_summary = data.get("split_turn_summary")
    if not isinstance(complete_turn_summary, str) or not complete_turn_summary.strip():
        raise RuntimeError("压缩摘要缺少 complete_turn_summary")
    if split_turn_summary is not None and not isinstance(split_turn_summary, str):
        raise RuntimeError("压缩摘要 split_turn_summary 必须是字符串或 null")

    return SummaryMessage(
        complete_turn_summary=complete_turn_summary,
        split_turn_summary=split_turn_summary,
    )


def print_banner(app_env: AppEnv, tool_registry: ToolRegistry, workspace_root: Path) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {app_env.openai_model}")
    print(f"base_url: {app_env.openai_base_url}")
    print(f"workspace: {workspace_root}")
    print(f"log_file: {LogWrapper.log_path()}")
    print(f"session_record: {SessionRecorder.session_path()}")
    print(f"terminal_log_level: {LogWrapper.filter_terminal_log_level()}")
    print(f"tools: {', '.join(tool_registry.names())}")
    print("输入 /context 查看当前内存里的 messages。")
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
        session_started_at = datetime.now()
        LogWrapper.configure(
            log_path=default_trace_log_path(),
            filter_terminal_log_level=app_env.filter_terminal_log_level,
            session_started_at=session_started_at,
        )
        SessionRecorder.configure(
            session_path=default_session_record_path(),
            session_started_at=session_started_at,
            metadata={
                "workspace": str(workspace_root),
                "model": app_env.openai_model,
                "base_url": app_env.openai_base_url,
            },
        )
        LogWrapper.cleanup_expired_logs_async()
        SessionRecorder.cleanup_expired_session_records_async()
    except ValueError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "示例：OPENAI_API_KEY=xxx OPENAI_MODEL=xxx python3 -m ai_job --workspace /path/to/project",
            file=sys.stderr,
        )
        return 2

    messages = build_initial_messages(app_env, workspace_root)
    SessionRecorder.record_session("SystemMessage", messages[0].content, "text")
    tool_registry = create_default_tool_registry(workspace_root, create_protected_grep_approval(workspace_root))
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
    print_banner(app_env, tool_registry, workspace_root)

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
        SessionRecorder.record_session("UserMessage", user_text, "text")
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
