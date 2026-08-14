"""Centralized environment loading for the ai_job app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env_file_loader import load_env_file


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = "90"
DEFAULT_MAX_TOOL_ROUNDS = "8"
DEFAULT_COMPACTION_RESERVE_TOKENS = "16384"
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = "20000"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful coding agent. Use tools when you need workspace information."
)
DEFAULT_FILTER_TERMINAL_LOG_LEVEL = "debug"


@dataclass(frozen=True)
class AppEnv:
    openai_api_key: str
    openai_model: str
    openai_base_url: str
    timeout_seconds: float
    max_tool_rounds: int
    context_window_override: int | None
    compaction_reserve_tokens: int
    compaction_keep_recent_tokens: int
    system_prompt: str
    filter_terminal_log_level: str


class EnvLoader:
    """Load .env and shell environment into a typed AppEnv object."""

    @classmethod
    def load(cls, env_file_path: Path) -> AppEnv:
        load_env_file(env_file_path)
        return cls.load_from_current_environment()

    @classmethod
    def load_from_current_environment(cls) -> AppEnv:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL")

        missing_names: list[str] = []
        if not openai_api_key:
            missing_names.append("OPENAI_API_KEY")
        if not openai_model:
            missing_names.append("OPENAI_MODEL")
        if missing_names:
            missing = ", ".join(missing_names)
            raise ValueError(f"缺少必要环境变量：{missing}")

        timeout_seconds = cls._read_positive_float(
            name="AI_JOB_TIMEOUT_SECONDS",
            default_value=DEFAULT_TIMEOUT_SECONDS,
        )
        max_tool_rounds = cls._read_positive_int(
            name="AI_JOB_MAX_TOOL_ROUNDS",
            default_value=DEFAULT_MAX_TOOL_ROUNDS,
        )
        context_window_override = cls._read_optional_positive_int("AI_JOB_CONTEXT_WINDOW")
        compaction_reserve_tokens = cls._read_positive_int(
            name="AI_JOB_COMPACTION_RESERVE_TOKENS",
            default_value=DEFAULT_COMPACTION_RESERVE_TOKENS,
        )
        compaction_keep_recent_tokens = cls._read_positive_int(
            name="AI_JOB_COMPACTION_KEEP_RECENT_TOKENS",
            default_value=DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        )

        return AppEnv(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/"),
            timeout_seconds=timeout_seconds,
            max_tool_rounds=max_tool_rounds,
            context_window_override=context_window_override,
            compaction_reserve_tokens=compaction_reserve_tokens,
            compaction_keep_recent_tokens=compaction_keep_recent_tokens,
            system_prompt=os.getenv("AI_JOB_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            filter_terminal_log_level=os.getenv(
                "FILTER_TERMINAL_LOG_LEVEL",
                DEFAULT_FILTER_TERMINAL_LOG_LEVEL,
            ),
        )

    @staticmethod
    def _read_positive_float(name: str, default_value: str) -> float:
        raw_value = os.getenv(name, default_value)
        try:
            parsed_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if parsed_value <= 0:
            raise ValueError(f"{name} 必须大于 0")
        return parsed_value

    @staticmethod
    def _read_optional_positive_int(name: str) -> int | None:
        raw_value = os.getenv(name)
        if raw_value is None or raw_value == "":
            return None
        try:
            parsed_value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if parsed_value <= 0:
            raise ValueError(f"{name} 必须大于 0")
        return parsed_value

    @staticmethod
    def _read_positive_int(name: str, default_value: str) -> int:
        raw_value = os.getenv(name, default_value)
        try:
            parsed_value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if parsed_value <= 0:
            raise ValueError(f"{name} 必须大于 0")
        return parsed_value
