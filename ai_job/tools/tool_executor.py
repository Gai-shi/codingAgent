"""Tool executor."""

from __future__ import annotations

from .tool_call import ToolCall
from .tool_registry import ToolRegistry


class ToolExecutor:
    """Execute internal ToolCall objects via a ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, tool_call: ToolCall) -> str:
        tool = self._registry.get(tool_call.name)
        if tool is None:
            return f"Error: unknown tool: {tool_call.name}"

        return tool.execute(tool_call.arguments)
