"""Terminal input mode that temporarily disables echo and discards pending input."""

from __future__ import annotations

from typing import Any

from .base import TerminalInputMode


class SuppressInputEchoAndDiscard(TerminalInputMode):
    """Hide terminal input while busy and discard typed characters on exit."""

    def _apply(self, attrs: list[Any], termios_module: Any) -> None:
        attrs[3] &= ~termios_module.ECHO

    def _before_restore(self, stdin_fd: int, termios_module: Any) -> None:
        termios_module.tcflush(stdin_fd, termios_module.TCIFLUSH)
