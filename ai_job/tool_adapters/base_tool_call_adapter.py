"""Base contracts for model tool-call adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..tools.tool_call import ToolCall
from ..tools.tool_registry import ToolRegistry


class BaseToolCallAdapter(ABC):
    """Interface contract between internal tools and provider-specific tool calling.

    子类负责把内部工具契约翻译成某个模型 API 的工具调用格式。基类只定义
    provider adapter 需要的工具协议转换能力，不负责 HTTP 请求或消息历史转换。
    """

    @abstractmethod
    def render_tool_definitions(self, tool_registry: ToolRegistry) -> list[dict[str, Any]]:
        """Render internal tool definitions into the model API's tool schema format."""

    @abstractmethod
    def render_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Render an internal ToolCall back into the provider's assistant message format."""

    @abstractmethod
    def get_tool_call_id(self, raw_tool_call: dict[str, Any]) -> str:
        """Extract the provider tool-call id needed to send a tool result back."""

    @abstractmethod
    def parse_tool_call(self, raw_tool_call: dict[str, Any]) -> ToolCall:
        """Convert one raw provider tool call into the agent-internal ToolCall."""
