"""OpenAI Chat Completions tool-call adapter.

这个模块只负责 OpenAI 原始格式和 agent 内部 ToolCall / ToolResult 之间的转换。
"""

from __future__ import annotations

import json
from typing import Any

from .tool_contract import ToolCall, ToolResult


def get_openai_tool_call_name(raw_tool_call: dict[str, Any]) -> str:
    """Best-effort tool name extraction for trace/debug output."""
    function_call = raw_tool_call.get("function")
    if not isinstance(function_call, dict):
        return "<malformed>"

    tool_name = function_call.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return "<malformed>"

    return tool_name


def parse_openai_tool_call(raw_tool_call: dict[str, Any]) -> ToolCall:
    """Convert one OpenAI Chat Completions tool_call object into internal ToolCall."""
    tool_call_id = raw_tool_call.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ValueError("malformed tool call: missing id")

    function_call = raw_tool_call.get("function")
    if not isinstance(function_call, dict):
        raise ValueError("malformed tool call: missing function object")

    tool_name = function_call.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError("malformed tool call: missing function.name")

    raw_arguments = function_call.get("arguments", "{}")
    if not isinstance(raw_arguments, str):
        raise ValueError("malformed tool call: function.arguments must be a JSON string")

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tool arguments JSON: {exc.msg}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("invalid tool arguments: expected a JSON object")

    return ToolCall(id=tool_call_id, name=tool_name, arguments=arguments)


def render_openai_tool_result_message(tool_call_id: str, tool_result: ToolResult) -> dict[str, Any]:
    """Render ToolResult as the role=tool message expected by Chat Completions."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": tool_result.content,
    }


def render_openai_tool_result(tool_call: ToolCall, tool_result: ToolResult) -> dict[str, Any]:
    return render_openai_tool_result_message(tool_call.id, tool_result)
