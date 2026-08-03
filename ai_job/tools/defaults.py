"""Default tool assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .grep import GrepTool
from .read_file import ReadFileTool
from .registry import ToolRegistry


def create_default_tool_registry(
    workspace_root: Path,
    request_protected_grep_approval: Optional[Callable[[Path], bool]] = None,
) -> ToolRegistry:
    return ToolRegistry(
        [
            ReadFileTool(workspace_root),
            GrepTool(workspace_root, request_protected_grep_approval),
        ]
    )
