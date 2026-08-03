"""OpenAI-compatible Chat Completions tool-call adapter."""

from __future__ import annotations

import copy
import json
from typing import Any

from ..tools.base_tool import BaseTool
from ..tools.tool_call import ToolCall
from ..tools.tool_registry import ToolRegistry
from .base_tool_call_adapter import BaseToolCallAdapter


class OpenAIToolCallAdapter(BaseToolCallAdapter):
    """Translate between internal tools and OpenAI-compatible tool calling."""

    def render_tool_definitions(self, tool_registry: ToolRegistry) -> list[dict[str, Any]]:
        return [self._render_tool_definition(tool) for tool in tool_registry.tools()]

    def _render_tool_definition(self, tool: BaseTool) -> dict[str, Any]:
        """Render one internal tool as an OpenAI Chat Completions tool schema."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": copy.deepcopy(tool.parameters_schema),
            },
        }

    def render_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Render an internal ToolCall as an OpenAI assistant.tool_calls item."""
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
            },
        }

    def get_tool_call_id(self, raw_tool_call: dict[str, Any]) -> str:
        tool_call_id = raw_tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("malformed tool call: missing id")
        return tool_call_id

    def parse_tool_call(self, raw_tool_call: dict[str, Any]) -> ToolCall:
        """Convert one OpenAI Chat Completions tool_call object into internal ToolCall."""
        tool_call_id = self.get_tool_call_id(raw_tool_call)

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
