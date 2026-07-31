"""A minimal terminal coding-agent loop with one native tool.

当前文件实现你已经拍板的边界：
- 只读环境变量；
- 非流式输出；
- 对话历史只保存在内存；
- 使用 OpenAI-compatible Chat Completions 接口；
- 使用原生 tool calling；
- 提供 read_file 和 grep 两个只读工具。
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional


EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding agent. Use tools when you need workspace information."
DEFAULT_MAX_TOOL_ROUNDS = 8
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_LOG_PATH = WORKSPACE_ROOT / ".ai_job" / "trace.log"
GREP_MAX_MATCHES = 50
GREP_MAX_LINE_CHARS = 300
DENIED_FILE_NAMES = {".env"}
DENIED_PATH_PARTS = {".ai_job", ".git", ".venv", "__pycache__"}


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


@dataclass(frozen=True)
class TraceLogger:
    log_path: Path
    print_to_terminal: bool

    @classmethod
    def from_env(cls) -> "TraceLogger":
        return cls(
            log_path=DEFAULT_TRACE_LOG_PATH,
            print_to_terminal=os.getenv("DebugMode", "true").strip().lower() == "true",
        )

    def write_round(self, round_number: int) -> None:
        self._write(f"round={round_number}")

    def write_tool(self, round_number: int, tool_name: str) -> None:
        self._write(f"round={round_number} tool={tool_name}")

    def _write(self, event_text: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {event_text}"

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as exc:
            raise RuntimeError(f"Trace 写入失败：{self.log_path}：{exc}") from exc

        if self.print_to_terminal:
            print(f"[trace] {line}", file=sys.stderr)


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

GREP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search UTF-8 text files in the workspace using a Python regular expression. "
            "Use this to locate relevant code before reading files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Optional directory path under the workspace root. "
                        "Defaults to the workspace root."
                    ),
                },
                "type": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional file extension filter without the dot, such as py, md. "
                        "Defaults to an empty string, which means searching all UTF-8 text files."
                    ),
                },
                "include_protected": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Whether to search hidden/protected directories. "
                        "Only use true after explicit user approval. Defaults to false."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}

AVAILABLE_TOOLS = [READ_FILE_TOOL, GREP_TOOL]


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


def resolve_workspace_directory(path_text: str) -> Path:
    """Resolve a path as a directory under WORKSPACE_ROOT."""
    raw_path = Path(path_text or ".").expanduser()
    candidate = raw_path if raw_path.is_absolute() else WORKSPACE_ROOT / raw_path
    resolved = candidate.resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path_text}") from exc

    if not resolved.exists():
        raise FileNotFoundError(f"directory not found: {path_text}")
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {path_text}")

    return resolved


def has_hidden_or_protected_dir(path: Path) -> bool:
    relative_parts = path.resolve().relative_to(WORKSPACE_ROOT).parts
    return any(part.startswith(".") or part in DENIED_PATH_PARTS for part in relative_parts)


def should_skip_directory(path: Path, allow_hidden_or_protected: bool) -> bool:
    if allow_hidden_or_protected:
        return False
    return path.name.startswith(".") or path.name in DENIED_PATH_PARTS


def normalize_file_type(type_value: Any) -> Optional[str]:
    if type_value is None:
        return None
    if not isinstance(type_value, str):
        raise ValueError('invalid arguments: "type" must be a string')

    normalized = type_value.strip().lstrip(".")
    if not normalized:
        return None
    if any(separator in normalized for separator in ("/", "\\")) or "*" in normalized:
        raise ValueError('invalid arguments: "type" must be a simple file extension, such as py')

    return normalized


def read_file_tool(arguments: dict[str, Any]) -> str:
    path_value = arguments.get("path")
    if not isinstance(path_value, str):
        raise ValueError('invalid arguments: "path" must be a string')

    file_path = resolve_workspace_file(path_value)
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {path_value}") from exc


@contextmanager
def allow_stdin_echo_for_input() -> Iterator[None]:
    """Temporarily allow visible terminal input inside a suppressed-input turn."""
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
        new_attrs[3] |= termios.ECHO
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, new_attrs)
    except (OSError, termios.error):
        yield
        return

    try:
        yield
    finally:
        with suppress(OSError, termios.error):
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)


def request_protected_grep_approval(path: Path) -> bool:
    relative_path = path.relative_to(WORKSPACE_ROOT) if path != WORKSPACE_ROOT else Path(".")
    print()
    print("grep 请求检索隐藏目录或保护目录。")
    print(f"范围：{relative_path}")
    with allow_stdin_echo_for_input():
        answer = input("如果你同意本次检索，请输入 yes；其它输入表示拒绝> ").strip().lower()
    return answer == "yes"


def grep_tool(arguments: dict[str, Any]) -> str:
    pattern_value = arguments.get("pattern")
    if not isinstance(pattern_value, str) or not pattern_value:
        raise ValueError('invalid arguments: "pattern" must be a non-empty string')

    path_value = arguments.get("path", ".")
    if not isinstance(path_value, str):
        raise ValueError('invalid arguments: "path" must be a string')

    include_protected = arguments.get("include_protected", False)
    if not isinstance(include_protected, bool):
        raise ValueError('invalid arguments: "include_protected" must be a boolean')

    type_filter = normalize_file_type(arguments.get("type"))

    try:
        regex = re.compile(pattern_value)
    except re.error as exc:
        raise ValueError(f"invalid regex pattern: {exc}") from exc

    search_root = resolve_workspace_directory(path_value)
    if has_hidden_or_protected_dir(search_root) and not include_protected:
        raise PermissionError(
            f"refusing to search hidden/protected directory without include_protected=true: {path_value}"
        )
    if include_protected and not request_protected_grep_approval(search_root):
        raise PermissionError("user rejected grep include_protected=true")

    matches: list[str] = []
    for current_dir, dir_names, file_names in os.walk(search_root):
        current_path = Path(current_dir)
        dir_names[:] = [
            name
            for name in sorted(dir_names)
            if not should_skip_directory(current_path / name, include_protected)
        ]

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            if file_path.name in DENIED_FILE_NAMES and not include_protected:
                continue
            if type_filter and file_path.suffix.lstrip(".") != type_filter:
                continue

            try:
                resolved_file = file_path.resolve()
                resolved_file.relative_to(WORKSPACE_ROOT)
            except (OSError, ValueError):
                continue
            if not resolved_file.is_file():
                continue

            try:
                with resolved_file.open("r", encoding="utf-8") as source_file:
                    for line_number, line in enumerate(source_file, start=1):
                        line_text = line.rstrip("\r\n")
                        if not regex.search(line_text):
                            continue

                        if len(line_text) > GREP_MAX_LINE_CHARS:
                            line_text = line_text[:GREP_MAX_LINE_CHARS] + "..."

                        relative_file = resolved_file.relative_to(WORKSPACE_ROOT)
                        matches.append(f"{relative_file}:{line_number}:{line_text}")
                        if len(matches) >= GREP_MAX_MATCHES:
                            matches.append(f"... truncated at {GREP_MAX_MATCHES} matches")
                            return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue

    if not matches:
        return "No matches."
    return "\n".join(matches)


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
        if tool_name == "grep":
            return grep_tool(arguments)
        return f"Error: unknown tool: {tool_name}"
    except Exception as exc:  # noqa: BLE001 - convert tool failures into LLM-readable text.
        return f"Error: {exc}"


def get_tool_call_name(tool_call: dict[str, Any]) -> str:
    function_call = tool_call.get("function")
    if not isinstance(function_call, dict):
        return "<malformed>"

    tool_name = function_call.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return "<malformed>"

    return tool_name


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


def print_banner(config: LLMConfig, trace_logger: TraceLogger) -> None:
    print("ai-job 最小 coding agent CLI")
    print(f"model: {config.model}")
    print(f"base_url: {config.base_url}")
    print(f"workspace: {WORKSPACE_ROOT}")
    print(f"trace_log: {trace_logger.log_path}")
    print("tools: read_file, grep")
    print("输入 /context 查看当前内存里的 messages。")
    print("输入 exit / quit / Ctrl-D 退出。")
    print()


def print_context(messages: list[dict[str, Any]]) -> None:
    print()
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()


@contextmanager
def suppress_stdin_echo_and_discard_input() -> Iterator[None]:
    """Hide and discard user typing while the CLI is busy."""
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


def run_agent_turn(config: LLMConfig, messages: list[dict[str, Any]], trace_logger: TraceLogger) -> str:
    """Run one user turn, including zero or more native tool-calling rounds."""
    for round_index in range(config.max_tool_rounds):
        round_number = round_index + 1
        trace_logger.write_round(round_number)

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
            trace_logger.write_tool(round_number, get_tool_call_name(tool_call))

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
    trace_logger = TraceLogger.from_env()
    print_banner(config, trace_logger)

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
                assistant_text = run_agent_turn(config, messages, trace_logger)
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
