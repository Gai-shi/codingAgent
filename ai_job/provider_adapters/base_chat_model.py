"""Base chat model contract for provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..communication import AssistantMessage, MessageHistory
from ..tools import ToolRegistry


class BaseChatModel(ABC):
    """Provider-independent chat model interface used by the agent loop."""

    @abstractmethod
    def complete(
        self,
        history: MessageHistory,
        tool_registry: ToolRegistry,
    ) -> AssistantMessage:
        """Call one non-streaming model completion and return an assistant message."""
