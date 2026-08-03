"""Base class for model-callable tools."""

from __future__ import annotations

import copy
from typing import Any

from .types import ToolResult


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
