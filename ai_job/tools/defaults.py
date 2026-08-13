"""Default tool assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .apply_patch_tool import ApplyPatchTool
from .compress_tool import CompressTool
from .grep_tool import GrepTool
from .read_file_tool import ReadFileTool
from .tool_registry import ToolRegistry


def create_default_tool_registry(
    workspace_root: Path,
    request_protected_grep_approval: Optional[Callable[[Path], bool]] = None,
    include_compress_tool: bool = True,
) -> ToolRegistry:
    tools = [
        ReadFileTool(workspace_root),
        GrepTool(workspace_root, request_protected_grep_approval),
        ApplyPatchTool(workspace_root),
    ]
    if include_compress_tool:
        tools.append(CompressTool())

    return ToolRegistry(tools)
