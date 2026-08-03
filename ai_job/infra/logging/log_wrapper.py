"""Small class-level logging facade.

LogWrapper 是当前项目的通用日志门面：
- 对外暴露 classmethod，调用方不需要实例化；
- 启动时可以通过 configure() 注入日志路径和终端镜像开关；
- debug/info/warn/error 四个方法保持同一调用形态。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import ClassVar


class LogWrapper:
    """A class-level logging facade with minimal global configuration."""

    _log_path: ClassVar[Path] = Path(".ai_job") / "trace.log"
    _print_to_terminal: ClassVar[bool] = True

    @classmethod
    def configure(cls, log_path: Path, print_to_terminal: bool) -> None:
        """Configure the shared logger.

        这是类级别配置，不需要调用方实例化 LogWrapper。
        """
        cls._log_path = log_path
        cls._print_to_terminal = print_to_terminal

    @classmethod
    def log_path(cls) -> Path:
        return cls._log_path

    @classmethod
    def debug(cls, tag: str, text: str) -> None:
        cls._write("DEBUG", tag, text)

    @classmethod
    def info(cls, tag: str, text: str) -> None:
        cls._write("INFO", tag, text)

    @classmethod
    def warn(cls, tag: str, text: str) -> None:
        cls._write("WARN", tag, text)

    @classmethod
    def error(cls, tag: str, text: str) -> None:
        cls._write("ERROR", tag, text)

    @classmethod
    def _write(cls, level: str, tag: str, text: str) -> None:
        cls._validate_string("tag", tag)
        cls._validate_string("text", text)

        timestamp = datetime.now().isoformat(timespec="seconds")
        safe_text = cls._single_line(text)
        line = f"{timestamp} {level} [{tag}] {safe_text}"

        try:
            cls._log_path.parent.mkdir(parents=True, exist_ok=True)
            with cls._log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as exc:
            raise RuntimeError(f"日志写入失败：{cls._log_path}：{exc}") from exc

        if cls._print_to_terminal:
            print(line, file=sys.stderr)

    @staticmethod
    def _validate_string(name: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} 必须是 str")

    @staticmethod
    def _single_line(text: str) -> str:
        return text.replace("\r", "\\r").replace("\n", "\\n")

