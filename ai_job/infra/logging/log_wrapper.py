"""Small class-level logging facade.

LogWrapper 是当前项目的通用日志门面：
- 对外暴露 classmethod，调用方不需要实例化；
- 启动时可以通过 configure() 注入日志基准路径和终端日志过滤等级；
- configure() 会把当前时间视为本次 CLI 会话开始时间，并派生本次会话的日志文件；
- debug/info/warn/error 四个方法保持同一调用形态。
"""

from __future__ import annotations

import calendar
import sys
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar, Optional


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
    _base_log_path: ClassVar[Path] = Path(".ai_job") / "logs" / "log.log"
    _log_path: ClassVar[Path] = Path(".ai_job") / "logs" / "log.log"
    _filter_terminal_log_level: ClassVar[LogLevel] = LogLevel.DEBUG

    @classmethod
    def configure(
        cls,
        log_path: Path,
        filter_terminal_log_level: str,
        session_started_at: Optional[datetime] = None,
    ) -> None:
        """Configure the shared logger.

        这是类级别配置，不需要调用方实例化 LogWrapper。
        log_path 是日志基准路径；真实写入路径会在文件名里追加本次会话开始时间。
        例如 .ai_job/logs/log.log 会写入 .ai_job/logs/log-20260805-103812-123.log。
        """
        started_at = session_started_at or cls._now()
        cls._base_log_path = log_path
        cls._log_path = cls._session_log_path(log_path, started_at)
        cls._filter_terminal_log_level = cls._normalize_filter_level(filter_terminal_log_level)

    @classmethod
    def log_path(cls) -> Path:
        return cls._log_path

    @classmethod
    def cleanup_expired_logs_async(cls) -> threading.Thread:
        """Start a best-effort background cleanup for session log files older than one month."""
        thread = threading.Thread(
            target=cls._cleanup_expired_logs_safely,
            name="ai-job-log-cleanup",
            daemon=True,
        )
        thread.start()
        return thread

    @classmethod
    def cleanup_expired_logs(cls) -> list[Path]:
        """Delete session log files whose filename timestamp is older than one calendar month.

        只清理当前日志基准路径派生出来的文件：
        - 基准路径 .ai_job/logs/log.log
        - 匹配文件 .ai_job/logs/log-YYYYMMDD-HHMMSS-mmm.log

        返回被删除的文件路径，方便测试和后续观测。
        """
        log_dir = cls._base_log_path.parent
        if not log_dir.exists():
            return []

        cutoff_time = cls._one_month_ago(cls._now())
        deleted_paths: list[Path] = []
        try:
            log_paths = list(log_dir.iterdir())
        except OSError as exc:
            raise RuntimeError(f"日志目录读取失败：{log_dir}：{exc}") from exc

        for log_path in log_paths:
            if not log_path.is_file():
                continue

            log_started_at = cls._session_log_started_at(log_path, cls._base_log_path)
            if log_started_at is None or log_started_at >= cutoff_time:
                continue

            try:
                log_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(f"过期日志删除失败：{log_path}：{exc}") from exc
            deleted_paths.append(log_path)

        return deleted_paths

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
        log_path = cls._log_path

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
        except OSError as exc:
            raise RuntimeError(f"日志写入失败：{log_path}：{exc}") from exc

        if cls._should_print_to_terminal(level):
            print(line, file=sys.stderr)

    @classmethod
    def _cleanup_expired_logs_safely(cls) -> None:
        try:
            cls.cleanup_expired_logs()
        except RuntimeError as exc:
            print(f"日志清理失败：{exc}", file=sys.stderr)

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @classmethod
    def _session_log_path(cls, base_log_path: Path, started_at: datetime) -> Path:
        return base_log_path.with_name(
            f"{base_log_path.stem}-{cls._session_time_text(started_at)}{base_log_path.suffix}"
        )

    @staticmethod
    def _session_time_text(started_at: datetime) -> str:
        millisecond = started_at.microsecond // 1000
        return f"{started_at.strftime('%Y%m%d-%H%M%S')}-{millisecond:03d}"

    @staticmethod
    def _session_log_started_at(log_path: Path, base_log_path: Path) -> Optional[datetime]:
        prefix = f"{base_log_path.stem}-"
        suffix = base_log_path.suffix
        file_name = log_path.name

        if not file_name.startswith(prefix):
            return None
        if suffix and not file_name.endswith(suffix):
            return None

        time_end = len(file_name) - len(suffix) if suffix else len(file_name)
        time_text = file_name[len(prefix):time_end]
        try:
            return datetime.strptime(time_text, "%Y%m%d-%H%M%S-%f")
        except ValueError:
            return None

    @staticmethod
    def _one_month_ago(moment: datetime) -> datetime:
        previous_month = moment.month - 1
        year = moment.year
        if previous_month == 0:
            previous_month = 12
            year -= 1

        last_day = calendar.monthrange(year, previous_month)[1]
        day = min(moment.day, last_day)
        return moment.replace(year=year, month=previous_month, day=day)

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
