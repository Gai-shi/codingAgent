"""Base class for model-callable tools."""

from __future__ import annotations

from typing import Any


class BaseTool:
    """Base class for all tools.

    子类只重写 _run()；execute() 保留给通用流程，例如异常包装和返回值校验。
    工具对 agent loop 只暴露字符串：成功返回文本，失败返回统一的 Error 文本。
    """

    name: str = ""
    description: str = ""
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def execute(self, arguments: dict[str, Any]) -> str:
        """Run the tool and convert failures into model-readable text."""
        try:
            self.validate_arguments(arguments)
            result = self._run(arguments)
            if not isinstance(result, str):
                raise TypeError(f"tool {self.name} returned non-string value")
            return result
        except Exception as exc:  # noqa: BLE001 - tool failures should be visible to the model.
            return f"Error: {exc}"

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate common argument shape.

        具体字段仍由子类或既有工具函数校验；第一版不引入完整 JSON Schema 校验器。
        """
        if not isinstance(arguments, dict):
            raise ValueError("invalid tool arguments: expected a JSON object")

    def _run(self, arguments: dict[str, Any]) -> str:
        raise NotImplementedError
