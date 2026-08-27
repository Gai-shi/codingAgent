"""Runner contract shared by hand-written and framework-based implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..communication import MessageState


class BaseAgentRunner(ABC):
    """Run one user turn, mutating message state history in-place."""

    @abstractmethod
    def run_turn(self, message_state: MessageState) -> str:
        """Run the agent loop for one user turn and return the final assistant text."""
