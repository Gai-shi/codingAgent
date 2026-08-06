"""Markdown session recording infrastructure."""

from __future__ import annotations

import calendar
import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Optional


class SessionRecorder:
    """Write one human-readable Markdown record per CLI session.

    SessionRecorder 不读取 FILTER_TERMINAL_LOG_LEVEL，也不输出到终端；
    它只负责把会话内容直接追加写入 sessions 文件。
    """

    _base_session_path: ClassVar[Path] = Path(".ai_job") / "sessions" / "sessions.md"
    _session_path: ClassVar[Path] = Path(".ai_job") / "sessions" / "sessions.md"
    _session_started_at: ClassVar[Optional[datetime]] = None
    _write_lock: ClassVar = threading.RLock()

    @classmethod
    def configure(
        cls,
        session_path: Path,
        session_started_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Configure the current CLI session record file.

        session_path 是会话记录基准路径；真实写入路径会在文件名里追加本次会话开始时间。
        例如 .ai_job/sessions/sessions.md 会写入
        .ai_job/sessions/sessions-20260805-103812-123.md。
        """
        started_at = session_started_at or cls._now()
        with cls._write_lock:
            cls._base_session_path = session_path
            cls._session_path = cls._session_record_path(session_path, started_at)
            cls._session_started_at = started_at
            cls._append_to_path_locked(cls._session_path, cls._render_session_header(started_at, metadata or {}))

    @classmethod
    def session_path(cls) -> Path:
        return cls._session_path

    @classmethod
    def record_text(cls, title: str, content: str) -> None:
        cls._validate_string("title", title)
        cls._validate_string("content", content)
        cls._write_section(title=title, content=content, language="text")

    @classmethod
    def record_json(cls, title: str, payload: Any) -> None:
        cls._validate_string("title", title)
        cls._write_section(
            title=title,
            content=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            language="json",
        )

    @classmethod
    def cleanup_expired_session_records_async(cls) -> threading.Thread:
        """Start a best-effort background cleanup for old session record files."""
        thread = threading.Thread(
            target=cls._cleanup_expired_session_records_safely,
            name="ai-job-session-record-cleanup",
            daemon=True,
        )
        thread.start()
        return thread

    @classmethod
    def cleanup_expired_session_records(cls) -> list[Path]:
        """Delete session records whose filename timestamp is older than one calendar month."""
        session_dir = cls._base_session_path.parent
        if not session_dir.exists():
            return []

        cutoff_time = cls._one_month_ago(cls._now())
        deleted_paths: list[Path] = []
        try:
            session_paths = list(session_dir.iterdir())
        except OSError as exc:
            raise RuntimeError(f"会话记录目录读取失败：{session_dir}：{exc}") from exc

        for session_path in session_paths:
            if not session_path.is_file():
                continue

            session_started_at = cls._session_record_started_at(session_path, cls._base_session_path)
            if session_started_at is None or session_started_at >= cutoff_time:
                continue

            try:
                session_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(f"过期会话记录删除失败：{session_path}：{exc}") from exc
            deleted_paths.append(session_path)

        return deleted_paths

    @classmethod
    def _write_section(cls, title: str, content: str, language: str) -> None:
        now = cls._now()
        text = cls._render_section(now, title, content, language)
        try:
            with cls._write_lock:
                cls._append_to_path_locked(cls._session_path, text)
        except OSError as exc:
            raise RuntimeError(f"会话记录写入失败：{cls._session_path}：{exc}") from exc

    @classmethod
    def _cleanup_expired_session_records_safely(cls) -> None:
        try:
            cls.cleanup_expired_session_records()
        except RuntimeError as exc:
            print(f"会话记录清理失败：{exc}", file=sys.stderr)

    @classmethod
    def _append_to_path_locked(cls, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as session_file:
            session_file.write(text)

    @classmethod
    def _render_session_header(cls, started_at: datetime, metadata: dict[str, Any]) -> str:
        lines = [
            f"# Session {cls._session_time_text(started_at)}",
            "",
            f"- started_at: `{started_at.isoformat(timespec='milliseconds')}`",
        ]
        for key, value in metadata.items():
            lines.append(f"- {key}: `{cls._single_line(cls._metadata_value_to_text(value))}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    @classmethod
    def _render_section(cls, now: datetime, title: str, content: str, language: str) -> str:
        lines = [
            f"## {now.strftime('%H:%M:%S')} {title}",
            "",
            cls._fenced_block(content, language),
            "",
        ]
        return "\n".join(lines) + "\n"

    @classmethod
    def _session_record_path(cls, base_session_path: Path, started_at: datetime) -> Path:
        return base_session_path.with_name(
            f"{base_session_path.stem}-{cls._session_time_text(started_at)}{base_session_path.suffix}"
        )

    @staticmethod
    def _session_time_text(started_at: datetime) -> str:
        millisecond = started_at.microsecond // 1000
        return f"{started_at.strftime('%Y%m%d-%H%M%S')}-{millisecond:03d}"

    @staticmethod
    def _session_record_started_at(session_path: Path, base_session_path: Path) -> Optional[datetime]:
        prefix = f"{base_session_path.stem}-"
        suffix = base_session_path.suffix
        file_name = session_path.name

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

    @staticmethod
    def _metadata_value_to_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, ensure_ascii=False, default=str)

    @classmethod
    def _fenced_block(cls, content: str, language: str) -> str:
        fence = cls._markdown_fence_for(content)
        return f"{fence}{language}\n{content}\n{fence}"

    @staticmethod
    def _markdown_fence_for(content: str) -> str:
        longest_run = 0
        current_run = 0
        for char in content:
            if char == "`":
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0
        return "`" * max(3, longest_run + 1)

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    @staticmethod
    def _validate_string(name: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} 必须是 str")

    @staticmethod
    def _single_line(text: str) -> str:
        return text.replace("\r", "\\r").replace("\n", "\\n")
