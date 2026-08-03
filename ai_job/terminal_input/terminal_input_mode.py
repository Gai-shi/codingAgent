"""Base context manager for temporary terminal input modes."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from types import TracebackType
from typing import Any, Optional, TextIO, Type


class TerminalInputMode:
    """Temporarily change stdin terminal attributes and restore them on exit.

    子类只负责描述“如何修改终端属性”；基类统一负责：
    - termios 不可用时降级为空操作；
    - stdin 不是 TTY 时降级为空操作；
    - 进入时保存旧状态；
    - 退出时恢复旧状态。
    """

    def __init__(self, stdin: Optional[TextIO] = None) -> None:
        self._stdin = stdin if stdin is not None else sys.stdin
        self._termios: Any = None
        self._stdin_fd: Optional[int] = None
        self._old_attrs: Optional[list[Any]] = None
        self._active = False

    def __enter__(self) -> "TerminalInputMode":
        try:
            import termios
        except ImportError:
            return self

        try:
            stdin_fd = self._stdin.fileno()
            if not os.isatty(stdin_fd):
                return self

            old_attrs = termios.tcgetattr(stdin_fd)
            new_attrs = old_attrs.copy()
            self._apply(new_attrs, termios)
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, new_attrs)
        except (OSError, termios.error):
            return self

        self._termios = termios
        self._stdin_fd = stdin_fd
        self._old_attrs = old_attrs
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        if not self._active or self._termios is None or self._stdin_fd is None or self._old_attrs is None:
            return False

        with suppress(OSError, self._termios.error):
            self._before_restore(self._stdin_fd, self._termios)
        with suppress(OSError, self._termios.error):
            self._termios.tcsetattr(self._stdin_fd, self._termios.TCSADRAIN, self._old_attrs)

        self._active = False
        return False

    def _apply(self, attrs: list[Any], termios_module: Any) -> None:
        raise NotImplementedError

    def _before_restore(self, stdin_fd: int, termios_module: Any) -> None:
        """Hook called before restoring original attributes."""
