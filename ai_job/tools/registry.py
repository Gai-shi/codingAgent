"""Tool registry."""

from __future__ import annotations

from typing import Optional

from .base import BaseTool


class ToolRegistry:
    """Registry that maps tool names to tool instances."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            if not tool.name:
                raise ValueError("tool name must not be empty")
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict]:
        return [tool.to_openai_schema() for tool in self._tools.values()]
