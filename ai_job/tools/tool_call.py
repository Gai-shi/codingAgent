"""Agent-internal tool call data structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by a model and executed by the agent."""

    id: str
    name: str
    arguments: dict[str, Any]
