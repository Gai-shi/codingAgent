"""A minimal terminal coding-agent loop with native tool calling.

当前文件实现你已经拍板的边界：
- 从项目级 .env 和环境变量读取配置；
- 非流式输出；
- 对话历史只保存在内存；
- 使用 OpenAI-compatible Chat Completions 接口；
- 使用原生 tool calling；
- 通过 ToolCall / ToolResult / BaseTool / ToolRegistry / ToolExecutor 管理工具；
- 提供 read_file 和 grep 两个只读工具。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import BaseToolCallAdapter, OpenAIToolCallAdapter
from .env_file_loader import load_env_file
from .infra.logging import LogWrapper
from .terminal_input import AllowInputEcho, SuppressInputEchoAndDiscard
from .tools import ToolExecutor, ToolRegistry, ToolResult, create_default_tool_registry


EXIT_COMMANDS = {"exit", "quit", "et", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding agent. Use tools when you need workspace information."
DEFAULT_MAX_TOOL_ROUNDS = 8
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE_PATH = WORKSPACE_ROOT / ".env"
DEFAULT_TRACE_LOG_PATH = WORKSPACE_ROOT / ".ai_job" / "trace.log"
TRACE_TAG = "trace"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float
    max_tool_rounds: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")

        missing_names: list[str] = []
        if not api_key:
            missing_names.append("OPENAI_API_KEY")
        if not model:
            missing_names.append("OPENAI_MODEL")
        if missing_names:
            missing = ", ".join(missing_names)
            raise ValueError(f"缺少必要环境变量：{missing}")

        timeout_raw = os.getenv("AI_JOB_TIMEOUT_SECONDS", "60")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ValueError("AI_JOB_TIMEOUT_SECONDS 必须是数字") from exc
        if timeout_seconds <= 0:
            raise ValueError("AI_JOB_TIMEOUT_SECONDS 必须大于 0")

        max_tool_rounds_raw = os.getenv("AI_JOB_MAX_TOOL_ROUNDS", str(DEFAULT_MAX_TOOL_ROUNDS))
        try:
            max_tool_rounds = int(max_tool_rounds_raw)
        except ValueError as exc:
            raise ValueError("AI_JOB_MAX_TOOL_ROUNDS 必须是整数") from exc
        if max_tool_rounds <= 0:
            raise ValueError("AI_JOB_MAX_TOOL_ROUNDS 必须大于 0")

        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_tool_rounds=max_tool_rounds,
        )


def request_protected_grep_approval(path: Path) -> bool:
    relative_path = path.relative_to(WORKSPACE_ROOT) if path != WORKSPACE_ROOT else Path(".")
    print()
    print("grep 请求检索隐藏目录或保护目录。")
    print(f"范围：{relative_path}")
    with AllowInputEcho():
        answer = input("如果你同意本次检索，请输入 yes；其它输入表示拒绝> ").strip().lower()
    return answer == "yes"


def call_llm(
    config: LLMConfig,
    messages: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    tool_call_adapter: BaseToolCallAdapter,
) -> dict[str, Any]:
    """Call one non-streaming chat completion and return the assistant message."""
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "tools": tool_call_adapter.render_tool_definitions(tool_registry),
        "tool_choice": "auto",
    }
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败：HTTP {exc.code}：{error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM 请求失败：{exc.reason}") from exc

    try:
        data: dict[str, Any] = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM 响应不是合法 JSON") from exc

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM 响应缺少 choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("LLM 响应 choices[0] 格式异常")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM 响应缺少 message")

    role = message.get("role")
    if role != "assistant":
        raise RuntimeError("LLM 响应 message.role 不是 assistant")

    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if content is not None and not isinstance(content, str):
        raise RuntimeError("LLM 响应 message.content 格式异常")
    if tool_calls is not None and not isinstance(tool_calls, list):
        raise RuntimeError("LLM 响应 message.tool_calls 格式异常")
    if content is None and not tool_calls:
        raise RuntimeError("LLM 响应既没有文本 content，也没有 tool_calls")

    return message


def build_initial_messages() -> list[dict[str, Any]]:
    system_prompt = os.getenv("AI_JOB_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    return [{"role": "system", "content": system_prompt}]


def print_banner(config: LLMConfig, tool_registry: ToolRegistry) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {config.model}")
    print(f"base_url: {config.base_url}")
    print(f"workspace: {WORKSPACE_ROOT}")
    print(f"trace_log: {LogWrapper.log_path()}")
    print(f"tools: {', '.join(tool_registry.names())}")
    print("输入 /context 查看当前内存里的 messages。")
    print("输入 exit / quit / et / Ctrl-D 退出。")
    print()


def print_context(messages: list[dict[str, Any]]) -> None:
    print()
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()


def run_agent_turn(
    config: LLMConfig,
    messages: list[dict[str, Any]],
    tool_registry: ToolRegistry,
    tool_executor: ToolExecutor,
    tool_call_adapter: BaseToolCallAdapter,
) -> str:
    """Run one user turn, including zero or more native tool-calling rounds."""
    for round_index in range(config.max_tool_rounds):
        round_number = round_index + 1
        LogWrapper.debug(TRACE_TAG, f"round={round_number}")

        assistant_message = call_llm(config, messages, tool_registry, tool_call_adapter)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            content = assistant_message.get("content")
            if not isinstance(content, str):
                raise RuntimeError("LLM 最终响应缺少文本 content")
            return content

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise RuntimeError("LLM 响应 tool_calls[] 格式异常")
            tool_name_for_trace = tool_call_adapter.get_tool_call_name_for_trace(tool_call)
            LogWrapper.debug(TRACE_TAG, f"round={round_number} tool={tool_name_for_trace}")

            try:
                tool_call_id = tool_call_adapter.get_tool_call_id(tool_call)
            except ValueError as exc:
                raise RuntimeError(f"LLM 响应 tool_call 格式异常：{exc}") from exc

            try:
                internal_tool_call = tool_call_adapter.parse_tool_call(tool_call)
            except ValueError as exc:
                tool_result = ToolResult(
                    ok=False,
                    content=f"Error: {exc}",
                    error_information=str(exc),
                )
                messages.append(
                    tool_call_adapter.render_tool_result_message(tool_call_id, tool_result)
                )
                continue

            tool_result = tool_executor.execute(internal_tool_call)
            messages.append(
                tool_call_adapter.render_tool_result_message(internal_tool_call.id, tool_result)
            )

    raise RuntimeError(f"工具调用轮数超过上限：{config.max_tool_rounds}")


def main() -> int:
    try:
        load_env_file(DEFAULT_ENV_FILE_PATH)
        config = LLMConfig.from_env()
    except ValueError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "示例：OPENAI_API_KEY=xxx OPENAI_MODEL=xxx python3 -m ai_job",
            file=sys.stderr,
        )
        return 2

    messages = build_initial_messages()
    LogWrapper.configure(
        log_path=DEFAULT_TRACE_LOG_PATH,
        print_to_terminal=os.getenv("DebugMode", "true").strip().lower() == "true",
    )
    tool_registry = create_default_tool_registry(WORKSPACE_ROOT, request_protected_grep_approval)
    tool_executor = ToolExecutor(tool_registry)
    tool_call_adapter = OpenAIToolCallAdapter()
    print_banner(config, tool_registry)

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
        messages.append({"role": "user", "content": user_text})
        try:
            with SuppressInputEchoAndDiscard():
                assistant_text = run_agent_turn(
                    config,
                    messages,
                    tool_registry,
                    tool_executor,
                    tool_call_adapter,
                )
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
