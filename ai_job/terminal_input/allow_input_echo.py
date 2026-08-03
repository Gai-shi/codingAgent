"""Terminal input mode that temporarily enables echo."""

from __future__ import annotations

from typing import Any

from .terminal_input_mode import TerminalInputMode


class AllowInputEcho(TerminalInputMode):
    """Temporarily allow visible terminal input."""

    def _apply(self, attrs: list[Any], termios_module: Any) -> None:
        attrs[3] |= termios_module.ECHO
