"""A minimal terminal coding-agent loop with one native tool.

当前文件实现你已经拍板的边界：
- 只读环境变量；
- 非流式输出；
- 对话历史只保存在内存；
- 使用 OpenAI-compatible Chat Completions 接口；
- 使用原生 tool calling；
- 第一版只提供 read_file 一个只读工具。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding agent. Use tools when you need workspace information."
DEFAULT_MAX_TOOL_ROUNDS = 8
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DENIED_FILE_NAMES = {".env"}
DENIED_PATH_PARTS = {".git", "__pycache__"}


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


READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file inside the current workspace. "
            "Use this when you need to inspect project files before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to the workspace root.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

AVAILABLE_TOOLS = [READ_FILE_TOOL]


def resolve_workspace_file(path_text: str) -> Path:
    """Resolve a user/model supplied path and keep it inside WORKSPACE_ROOT."""
    if not path_text:
        raise ValueError("missing required argument: path")

    raw_path = Path(path_text).expanduser()
    candidate = raw_path if raw_path.is_absolute() else WORKSPACE_ROOT / raw_path
    resolved = candidate.resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path_text}") from exc

    relative_parts = resolved.relative_to(WORKSPACE_ROOT).parts
    if resolved.name in DENIED_FILE_NAMES or any(part in DENIED_PATH_PARTS for part in relative_parts):
        raise PermissionError(f"refusing to read protected path: {path_text}")

    if not resolved.exists():
        raise FileNotFoundError(f"file not found: {path_text}")
    if not resolved.is_file():
        raise ValueError(f"not a file: {path_text}")

    return resolved


def read_file_tool(arguments: dict[str, Any]) -> str:
    path_value = arguments.get("path")
    if not isinstance(path_value, str):
        raise ValueError('invalid arguments: "path" must be a string')

    file_path = resolve_workspace_file(path_value)
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {path_value}") from exc


def execute_tool_call(tool_call: dict[str, Any]) -> str:
    """Execute one Chat Completions tool_call and return plain string content."""
    function_call = tool_call.get("function")
    if not isinstance(function_call, dict):
        return "Error: malformed tool call: missing function object"

    tool_name = function_call.get("name")
    raw_arguments = function_call.get("arguments", "{}")
    if not isinstance(raw_arguments, str):
        return "Error: malformed tool call: function.arguments must be a JSON string"

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return f"Error: invalid tool arguments JSON: {exc.msg}"
    if not isinstance(arguments, dict):
        return "Error: invalid tool arguments: expected a JSON object"

    try:
        if tool_name == "read_file":
            return read_file_tool(arguments)
        return f"Error: unknown tool: {tool_name}"
    except Exception as exc:  # noqa: BLE001 - convert tool failures into LLM-readable text.
        return f"Error: {exc}"


def call_llm(config: LLMConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Call one non-streaming chat completion and return the assistant message."""
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "tools": AVAILABLE_TOOLS,
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


def print_banner(config: LLMConfig) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {config.model}")
    print(f"base_url: {config.base_url}")
    print(f"workspace: {WORKSPACE_ROOT}")
    print("tools: read_file")
    print("输入 /context 查看当前内存里的 messages。")
    print("输入 exit / quit / Ctrl-D 退出。")
    print()


def print_context(messages: list[dict[str, Any]]) -> None:
    print()
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()


@contextmanager
def suppress_stdin_echo_and_discard_input() -> Iterator[None]:
    """Hide and discard user typing while the CLI is waiting for the model."""
    try:
        import termios
    except ImportError:
        yield
        return

    try:
        stdin_fd = sys.stdin.fileno()
        if not os.isatty(stdin_fd):
            yield
            return

        old_attrs = termios.tcgetattr(stdin_fd)
        new_attrs = old_attrs.copy()
        new_attrs[3] &= ~termios.ECHO
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, new_attrs)
    except (OSError, termios.error):
        yield
        return

    try:
        yield
    finally:
        with suppress(OSError, termios.error):
            termios.tcflush(stdin_fd, termios.TCIFLUSH)
        with suppress(OSError, termios.error):
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)


def run_agent_turn(config: LLMConfig, messages: list[dict[str, Any]]) -> str:
    """Run one user turn, including zero or more native tool-calling rounds."""
    for _round_index in range(config.max_tool_rounds):
        assistant_message = call_llm(config, messages)
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
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_call_id, str):
                raise RuntimeError("LLM 响应 tool_call 缺少 id")

            tool_result_text = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result_text,
                }
            )

    raise RuntimeError(f"工具调用轮数超过上限：{config.max_tool_rounds}")


def main() -> int:
    try:
        config = LLMConfig.from_env()
    except ValueError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        print(
            "示例：OPENAI_API_KEY=xxx OPENAI_MODEL=xxx python3 -m ai_job.chat_cli",
            file=sys.stderr,
        )
        return 2

    messages = build_initial_messages()
    print_banner(config)

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
            with suppress_stdin_echo_and_discard_input():
                assistant_text = run_agent_turn(config, messages)
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
