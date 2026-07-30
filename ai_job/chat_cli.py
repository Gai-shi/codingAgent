"""A minimal non-streaming terminal chat loop.

当前文件只实现你已经拍板的第一版边界：
- 只读环境变量；
- 非流式输出；
- 对话历史只保存在内存；
- 只走一个 OpenAI-compatible Chat Completions 风格的模型接口。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
CONTEXT_COMMANDS = {"/context"}
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float

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

        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )


def call_llm(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    """Call one non-streaming chat completion and return assistant text."""
    url = f"{config.base_url}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
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

    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM 响应缺少文本 content")

    return content


def build_initial_messages() -> list[dict[str, str]]:
    system_prompt = os.getenv("AI_JOB_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    return [{"role": "system", "content": system_prompt}]


def print_banner(config: LLMConfig) -> None:
    print("ai-job 最小聊天 CLI")
    print(f"model: {config.model}")
    print(f"base_url: {config.base_url}")
    print("输入 /context 查看当前内存里的 messages。")
    print("输入 exit / quit / Ctrl-D 退出。")
    print()


def print_context(messages: list[dict[str, str]]) -> None:
    print()
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print()


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

        messages.append({"role": "user", "content": user_text})
        try:
            assistant_text = call_llm(config, messages)
        except RuntimeError as exc:
            messages.pop()
            print(f"错误：{exc}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": assistant_text})
        print(f"\n助手> {assistant_text}\n")


if __name__ == "__main__":
    raise SystemExit(main())
