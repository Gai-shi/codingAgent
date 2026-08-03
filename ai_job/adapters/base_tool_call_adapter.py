"""Base contracts for model tool-call adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..tools.tool_registry import ToolRegistry
from ..tools.types import ToolCall, ToolResult


class BaseToolCallAdapter(ABC):
    """Interface contract between the agent loop and provider-specific tool calling.

    子类负责把内部工具契约翻译成某个模型 API 的工具调用格式。基类只定义
    agent loop 需要的能力，不预设 OpenAI、Anthropic 或其它 provider 的字段结构。
    """

    @abstractmethod
    def render_tool_definitions(self, tool_registry: ToolRegistry) -> list[dict[str, Any]]:
        """Render internal tool definitions into the model API's tool schema format."""

    @abstractmethod
    def get_tool_call_name_for_trace(self, raw_tool_call: dict[str, Any]) -> str:
        """Best-effort extract the tool name from a raw provider tool call for tracing."""

    @abstractmethod
    def get_tool_call_id(self, raw_tool_call: dict[str, Any]) -> str:
        """Extract the provider tool-call id needed to send a tool result back."""

    @abstractmethod
    def parse_tool_call(self, raw_tool_call: dict[str, Any]) -> ToolCall:
        """Convert one raw provider tool call into the agent-internal ToolCall."""

    @abstractmethod
    def render_tool_result_message(
        self,
        tool_call_id: str,
        tool_result: ToolResult,
    ) -> dict[str, Any]:
        """Render a ToolResult into the model API's tool-result message format."""
