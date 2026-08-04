"""OpenAI-compatible Chat Completions provider adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

from ..communication import (
    AssistantMessage,
    Message,
    MessageHistory,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ..infra.env import AppEnv
from ..tool_adapters import BaseToolCallAdapter, OpenAIToolCallAdapter
from ..tools import ToolRegistry
from .base_chat_model import BaseChatModel


class OpenAIModel(BaseChatModel):
    """Non-streaming OpenAI-compatible Chat Completions model adapter."""

    def __init__(
        self,
        app_env: AppEnv,
        tool_call_adapter: Optional[BaseToolCallAdapter] = None,
    ) -> None:
        self._app_env = app_env
        self._tool_call_adapter = tool_call_adapter or OpenAIToolCallAdapter()

    def complete(
        self,
        history: MessageHistory,
        tool_registry: ToolRegistry,
    ) -> AssistantMessage:
        """Call one non-streaming chat completion and return a normalized message."""
        response_body = self._request_completion(history, tool_registry)
        return self._parse_assistant_message(response_body)

    def _request_completion(
        self,
        history: MessageHistory,
        tool_registry: ToolRegistry,
    ) -> str:
        url = f"{self._app_env.openai_base_url}/chat/completions"
        payload = {
            "model": self._app_env.openai_model,
            "messages": [self._render_message(message) for message in history],
            "tools": self._tool_call_adapter.render_tool_definitions(tool_registry),
            "tool_choice": "auto",
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._app_env.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._app_env.timeout_seconds,
            ) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM 请求失败：HTTP {exc.code}：{error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 请求失败：{exc.reason}") from exc

    def _render_message(self, message: Message) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        if isinstance(message, UserMessage):
            return {"role": "user", "content": message.content}
        if isinstance(message, AssistantMessage):
            rendered: dict[str, Any] = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                rendered["tool_calls"] = [
                    self._tool_call_adapter.render_tool_call(tool_call)
                    for tool_call in message.tool_calls
                ]
            return rendered
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            }

        raise TypeError(f"unknown message type: {type(message).__name__}")

    def _parse_assistant_message(self, response_body: str) -> AssistantMessage:
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

        raw_message = first_choice.get("message")
        if not isinstance(raw_message, dict):
            raise RuntimeError("LLM 响应缺少 message")

        role = raw_message.get("role")
        if role != "assistant":
            raise RuntimeError("LLM 响应 message.role 不是 assistant")

        content = raw_message.get("content")
        raw_tool_calls = raw_message.get("tool_calls")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("LLM 响应 message.content 格式异常")
        if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
            raise RuntimeError("LLM 响应 message.tool_calls 格式异常")
        if content is None and not raw_tool_calls:
            raise RuntimeError("LLM 响应既没有文本 content，也没有 tool_calls")

        tool_calls = []
        for raw_tool_call in raw_tool_calls or []:
            if not isinstance(raw_tool_call, dict):
                raise RuntimeError("LLM 响应 tool_calls[] 格式异常")
            try:
                tool_calls.append(self._tool_call_adapter.parse_tool_call(raw_tool_call))
            except ValueError as exc:
                raise RuntimeError(f"LLM 响应 tool_call 格式异常：{exc}") from exc

        return AssistantMessage(content=content, tool_calls=tool_calls)
