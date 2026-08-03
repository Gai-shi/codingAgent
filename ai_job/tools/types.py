"""Shared tool data structures."""

from __future__ import annotations

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
