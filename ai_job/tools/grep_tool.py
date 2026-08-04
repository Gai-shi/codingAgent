"""grep tool."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .base_tool import BaseTool
from .path_policy import (
    DENIED_FILE_NAMES,
    is_hidden_or_protected_dir,
    resolve_workspace_directory,
    should_skip_directory,
)


GREP_MAX_MATCHES = 50
GREP_MAX_LINE_CHARS = 300
GREP_DESCRIPTION = (
    "Search UTF-8 text files in the workspace using a Python regular expression. "
    "Use this to locate relevant code before reading files."
)
GREP_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Python regular expression pattern to search for.",
        },
        "path": {
            "type": "string",
            "description": (
                "Optional directory path under the workspace root. "
                "Defaults to the workspace root."
            ),
        },
        "type": {
            "type": "string",
            "default": "",
            "description": (
                "Optional file extension filter without the dot, such as py, md. "
                "Defaults to an empty string, which means searching all UTF-8 text files."
            ),
        },
        "include_protected": {
            "type": "boolean",
            "default": False,
            "description": (
                "Whether to search hidden/protected directories. "
                "Only use true after explicit user approval. Defaults to false."
            ),
        },
    },
    "required": ["pattern"],
    "additionalProperties": False,
}


def normalize_file_type(type_value: Any) -> Optional[str]:
    if type_value is None:
        return None
    if not isinstance(type_value, str):
        raise ValueError('invalid arguments: "type" must be a string')

    normalized = type_value.strip().lstrip(".")
    if not normalized:
        return None
    if any(separator in normalized for separator in ("/", "\\")) or "*" in normalized:
        raise ValueError('invalid arguments: "type" must be a simple file extension, such as py')

    return normalized


def grep_text(
    arguments: dict[str, Any],
    workspace_root: Path,
    request_protected_approval: Optional[Callable[[Path], bool]] = None,
) -> str:
    pattern_value = arguments.get("pattern")
    if not isinstance(pattern_value, str) or not pattern_value:
        raise ValueError('invalid arguments: "pattern" must be a non-empty string')

    path_value = arguments.get("path", ".")
    if not isinstance(path_value, str):
        raise ValueError('invalid arguments: "path" must be a string')

    include_protected = arguments.get("include_protected", False)
    if not isinstance(include_protected, bool):
        raise ValueError('invalid arguments: "include_protected" must be a boolean')

    type_filter = normalize_file_type(arguments.get("type"))

    try:
        regex = re.compile(pattern_value)
    except re.error as exc:
        raise ValueError(f"invalid regex pattern: {exc}") from exc

    search_root = resolve_workspace_directory(path_value, workspace_root)
    if is_hidden_or_protected_dir(search_root, workspace_root) and not include_protected:
        raise PermissionError(
            f"refusing to search hidden/protected directory without include_protected=true: {path_value}"
        )
    if include_protected:
        if request_protected_approval is None:
            raise PermissionError("grep include_protected=true requires an approval callback")
        if not request_protected_approval(search_root):
            raise PermissionError("user rejected grep include_protected=true")

    matches: list[str] = []
    for current_dir, dir_names, file_names in os.walk(search_root):
        current_path = Path(current_dir)
        dir_names[:] = [
            name
            for name in sorted(dir_names)
            if not should_skip_directory(current_path / name, include_protected)
        ]

        for file_name in sorted(file_names):
            file_path = current_path / file_name
            if file_path.name in DENIED_FILE_NAMES and not include_protected:
                continue
            if type_filter and file_path.suffix.lstrip(".") != type_filter:
                continue

            try:
                resolved_file = file_path.resolve()
                resolved_file.relative_to(workspace_root)
            except (OSError, ValueError):
                continue
            if not resolved_file.is_file():
                continue

            try:
                with resolved_file.open("r", encoding="utf-8") as source_file:
                    for line_number, line in enumerate(source_file, start=1):
                        line_text = line.rstrip("\r\n")
                        if not regex.search(line_text):
                            continue

                        if len(line_text) > GREP_MAX_LINE_CHARS:
                            line_text = line_text[:GREP_MAX_LINE_CHARS] + "..."

                        relative_file = resolved_file.relative_to(workspace_root)
                        matches.append(f"{relative_file}:{line_number}:{line_text}")
                        if len(matches) >= GREP_MAX_MATCHES:
                            matches.append(f"... truncated at {GREP_MAX_MATCHES} matches")
                            return "\n".join(matches)
            except (OSError, UnicodeDecodeError):
                continue

    if not matches:
        return "No matches."
    return "\n".join(matches)


class GrepTool(BaseTool):
    name = "grep"
    description = GREP_DESCRIPTION
    parameters_schema = GREP_PARAMETERS_SCHEMA

    def __init__(
        self,
        workspace_root: Path,
        request_protected_approval: Optional[Callable[[Path], bool]] = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._request_protected_approval = request_protected_approval

    def _run(self, arguments: dict[str, Any]) -> str:
        return grep_text(arguments, self._workspace_root, self._request_protected_approval)
