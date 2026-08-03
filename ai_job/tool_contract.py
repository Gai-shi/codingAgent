"""Internal tool contract for the learning coding agent.

这个模块只描述 agent 内部的工具抽象，不关心 OpenAI / Anthropic 等外部协议。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ToolResult:
    """Standard result returned by every tool.

    content 永远是回填给模型看的字符串；error_information 主要给调试和日志使用。
    """

    ok: bool
    content: str
    error_information: Optional[str] = None


@dataclass(frozen=True)
class ToolCall:
    """Agent-internal representation of one tool call requested by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


class BaseTool:
    """Base class for all tools.

    子类只重写 _run()；execute() 保留给通用流程，例如异常包装和返回值校验。
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool and convert failures into ToolResult."""
        try:
            self.validate_arguments(arguments)
            result = self._run(arguments)
            if not isinstance(result, ToolResult):
                raise TypeError(f"tool {self.name} returned non-ToolResult value")
            return result
        except Exception as exc:  # noqa: BLE001 - tool failures should be visible to the model.
            error_text = str(exc)
            return ToolResult(
                ok=False,
                content=f"Error: {error_text}",
                error_information=error_text,
            )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate common argument shape.

        具体字段仍由子类或既有工具函数校验；第一版不引入完整 JSON Schema 校验器。
        """
        if not isinstance(arguments, dict):
            raise ValueError("invalid tool arguments: expected a JSON object")

    def _run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this internal tool definition as an OpenAI Chat Completions tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(self.parameters_schema),
            },
        }


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

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]


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
