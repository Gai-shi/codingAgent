"""Small class-level logging facade.

LogWrapper 是当前项目的通用日志门面：
- 对外暴露 classmethod，调用方不需要实例化；
- 启动时可以通过 configure() 注入日志基准路径和终端日志过滤等级；
- 每次写入时按当前日期派生真实日志文件，做到每天一个日志文件；
- debug/info/warn/error 四个方法保持同一调用形态。
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    NONE = "none"


class LogWrapper:
    """A class-level logging facade with minimal global configuration."""

    _LEVEL_RANK: ClassVar[dict[LogLevel, int]] = {
        LogLevel.DEBUG: 10,
        LogLevel.INFO: 20,
        LogLevel.WARN: 30,
        LogLevel.ERROR: 40,
        LogLevel.NONE: 50,
    }
    _base_log_path: ClassVar[Path] = Path(".ai_job") / "trace.log"
    _filter_terminal_log_level: ClassVar[LogLevel] = LogLevel.DEBUG

    @classmethod
    def configure(cls, log_path: Path, filter_terminal_log_level: str) -> None:
        """Configure the shared logger.

        这是类级别配置，不需要调用方实例化 LogWrapper。
        log_path 是日志基准路径；真实写入路径会在文件名里追加当天日期。
        例如 .ai_job/trace.log 会写入 .ai_job/trace-2026-08-05.log。
        """
        cls._base_log_path = log_path
        cls._filter_terminal_log_level = cls._normalize_filter_level(filter_terminal_log_level)

    @classmethod
    def log_path(cls) -> Path:
        return cls._daily_log_path(cls._base_log_path, cls._now().date().isoformat())

    @classmethod
    def filter_terminal_log_level(cls) -> str:
        return cls._filter_terminal_log_level.value

    @classmethod
    def debug(cls, tag: str, text: str) -> None:
        cls._write(LogLevel.DEBUG, tag, text)

    @classmethod
    def info(cls, tag: str, text: str) -> None:
        cls._write(LogLevel.INFO, tag, text)

    @classmethod
    def warn(cls, tag: str, text: str) -> None:
        cls._write(LogLevel.WARN, tag, text)

    @classmethod
    def error(cls, tag: str, text: str) -> None:
        cls._write(LogLevel.ERROR, tag, text)

    @classmethod
    def _write(cls, level: LogLevel, tag: str, text: str) -> None:
        cls._validate_string("tag", tag)
        cls._validate_string("text", text)

        now = cls._now()
        timestamp = now.isoformat(timespec="seconds")
        safe_text = cls._single_line(text)
        line = f"{timestamp} {level.value.upper()} [{tag}] {safe_text}"
        log_path = cls._daily_log_path(cls._base_log_path, now.date().isoformat())

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as exc:
            raise RuntimeError(f"日志写入失败：{log_path}：{exc}") from exc

        if cls._should_print_to_terminal(level):
            print(line, file=sys.stderr)

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _daily_log_path(base_log_path: Path, date_text: str) -> Path:
        return base_log_path.with_name(f"{base_log_path.stem}-{date_text}{base_log_path.suffix}")

    @classmethod
    def _should_print_to_terminal(cls, level: LogLevel) -> bool:
        return cls._LEVEL_RANK[level] >= cls._LEVEL_RANK[cls._filter_terminal_log_level]

    @classmethod
    def _normalize_filter_level(cls, filter_terminal_log_level: str) -> LogLevel:
        cls._validate_string("filter_terminal_log_level", filter_terminal_log_level)

        normalized = filter_terminal_log_level.strip().lower()
        try:
            return LogLevel(normalized)
        except ValueError as exc:
            allowed_levels = ", ".join(level.value for level in LogLevel)
            raise ValueError(
                "FILTER_TERMINAL_LOG_LEVEL 必须是以下值之一："
                f"{allowed_levels}，当前值：{filter_terminal_log_level}"
            ) from exc

    @staticmethod
    def _validate_string(name: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} 必须是 str")

    @staticmethod
    def _single_line(text: str) -> str:
        return text.replace("\r", "\\r").replace("\n", "\\n")
