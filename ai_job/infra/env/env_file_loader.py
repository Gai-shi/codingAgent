"""Minimal project-level .env loader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(env_file_path: Path) -> None:
    """Load KEY=VALUE pairs from env_file_path without overriding existing env vars.

    这是项目级轻量 loader，不追求完整兼容所有 dotenv 语法；当前支持：
    - 空行和以 # 开头的注释行；
    - KEY=VALUE；
    - export KEY=VALUE；
    - 单引号或双引号包裹的 VALUE。
    """
    if not env_file_path.exists():
        return
    if not env_file_path.is_file():
        raise ValueError(f".env 路径不是文件：{env_file_path}")

    try:
        lines = env_file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"读取 .env 失败：{env_file_path}：{exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f".env 不是合法 UTF-8 文本：{env_file_path}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        parsed = _parse_env_line(raw_line, line_number)
        if parsed is None:
            continue

        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_env_line(raw_line: str, line_number: int) -> Optional[tuple[str, str]]:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].strip()

    if "=" not in line:
        raise ValueError(f".env 第 {line_number} 行格式错误：缺少 =")

    key, value = line.split("=", 1)
    key = key.strip()
    if not ENV_KEY_PATTERN.fullmatch(key):
        raise ValueError(f".env 第 {line_number} 行变量名非法：{key!r}")

    return key, _normalize_env_value(value)


def _normalize_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
