"""read_file tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base_tool import BaseTool
from .path_policy import resolve_workspace_file
from .types import ToolResult


READ_FILE_DESCRIPTION = (
    "Read a UTF-8 text file inside the current workspace. "
    "Use this when you need to inspect project files before answering."
)
READ_FILE_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file, relative to the workspace root.",
        }
    },
    "required": ["path"],
    "additionalProperties": False,
}


def read_file_text(arguments: dict[str, Any], workspace_root: Path) -> str:
    path_value = arguments.get("path")
    if not isinstance(path_value, str):
        raise ValueError('invalid arguments: "path" must be a string')

    file_path = resolve_workspace_file(path_value, workspace_root)
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {path_value}") from exc


class ReadFileTool(BaseTool):
    name = "read_file"
    description = READ_FILE_DESCRIPTION
    parameters_schema = READ_FILE_PARAMETERS_SCHEMA

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    def _run(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=read_file_text(arguments, self._workspace_root))
