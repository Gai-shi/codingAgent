"""Tool executor."""

from __future__ import annotations

from .tool_registry import ToolRegistry
from .types import ToolCall, ToolResult


class ToolExecutor:
    """Execute internal ToolCall objects via a ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self._registry.get(tool_call.name)
        if tool is None:
            error_text = f"unknown tool: {tool_call.name}"
            return ToolResult(
                ok=False,
                content=f"Error: {error_text}",
                error_information=error_text,
            )

        return tool.execute(tool_call.arguments)
